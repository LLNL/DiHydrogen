#include "distconv/dnn_backend/mean_squared_error.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

#include <limits>

#if H2_HAS_CUDA
#include <cub/block/block_reduce.cuh>
namespace cubns = cub;
#elif H2_HAS_ROCM
#include <hipcub/block/block_reduce.hpp>
namespace cubns = hipcub;
#endif

using distconv::tensor::LocaleMPI;
using distconv::tensor::CUDAAllocator;

template <typename DataType>
using TensorCUDA = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;
using MeanSquaredErrorCUDNN =
    distconv::MeanSquaredError<distconv::BackendDNNLib>;

namespace distconv {
namespace mean_squared_error {

/*
  - gridDim.y == number of samples
  - Each sample is taken care by gridDim.x blocks
*/
template <typename DataType, int BLOCK_SIZE>
__global__ void fp_local(const DataType * __restrict__ prediction,
                         const DataType * __restrict__ ground_truth,
                         DataType * __restrict__ y,
                         const index_t sample_size,
                         const index_t sample_spatial_size,
                         const index_t sample_channel_size,
                         int thread_work_size) {
  const int tid = threadIdx.x;
  const int sample_idx = blockIdx.y;

  prediction += sample_idx * sample_size;
  ground_truth += sample_idx * sample_size;

  index_t offset = tid + blockIdx.x * BLOCK_SIZE;
  const int offset_stride = BLOCK_SIZE * gridDim.x;
  const index_t offset_limit = min(
      sample_size, offset + offset_stride * thread_work_size);

  auto psum = DataType(0.);
  for (; offset < offset_limit; offset += offset_stride) {
    const DataType x = prediction[offset];
    const DataType xhat = ground_truth[offset];
    const DataType err = x - xhat;
    psum += err * err;
  }

  using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  psum = BlockReduce(temp_storage).Sum(psum) / sample_size;

  if (tid == 0) {
    atomic_add(&y[sample_idx], psum);
  }
}


/*
  - gridDim.y == number of samples
  - Each sample is taken care by gridDim.x blocks
*/
template <typename DataType, int BLOCK_SIZE>
__global__ void bp_local(const DataType * __restrict__ x_pred,
                         const DataType * __restrict__ x_truth,
                         const DataType * __restrict__ dy,
                         DataType * __restrict__ dx_pred,
                         DataType * __restrict__ dx_truth,
                         const index_t sample_size,
                         const index_t sample_spatial_size,
                         const index_t sample_channel_size,
                         int thread_work_size) {
  const int tid = threadIdx.x;
  const int sample_idx = blockIdx.y;

  x_pred += sample_idx * sample_size;
  dx_pred += sample_idx * sample_size;
  x_truth += sample_idx * sample_size;
  dx_truth += sample_idx * sample_size;

  index_t offset = tid + blockIdx.x * BLOCK_SIZE;
  const int offset_stride = BLOCK_SIZE * gridDim.x;
  const index_t offset_limit = min(
      sample_size, offset + offset_stride * thread_work_size);

  const auto dy_sample = dy[sample_idx];
  const DataType scale = static_cast<DataType>(DataType(2) / sample_size);
  for (; offset < offset_limit; offset += offset_stride) {
    const DataType x = x_pred[offset];
    const DataType xhat = x_truth[offset];
    const DataType err = x - xhat;
    dx_pred[offset] = scale * err * dy_sample;
    dx_truth[offset] = - scale * err * dy_sample;
  }
}

} // namespace mean_squared_error

template <typename Tensor>
int MeanSquaredErrorCUDNN::forward(const Tensor &x_pred, const Tensor &x_truth,
                                   Tensor &y) {
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
      << "Mean squared error FP: " << x_pred << ", "
      << x_truth << ", " << y;

  constexpr int block_size = 256;
  constexpr int thread_work_size = 8;

  // Assumes no halo for simplicity
  assert_eq(x_pred.get_local_size(), x_pred.get_local_real_size());
  assert_eq(x_truth.get_local_size(), x_truth.get_local_real_size());

  const auto num_samples = x_pred.get_local_shape()[-1];

  if (num_samples == 0) return 0;

  y.zero(m_be.get_stream());

  if (x_pred.get_local_size() > 0) {
    auto sample_size = x_pred.get_local_size() / num_samples;
    auto num_blocks_per_sample = util::ceil(
        sample_size, (index_t)block_size * thread_work_size);

    dim3 bdim(block_size);
    dim3 gdim(num_blocks_per_sample, num_samples);

    const auto sample_channel_size = x_pred.get_local_shape()[x_pred.get_num_spatial_dims()];
    const auto sample_spatial_size = sample_size / sample_channel_size;
    assert_eq(sample_channel_size*sample_spatial_size, sample_size);

    mean_squared_error::fp_local<DataType, block_size>
        <<<gdim, bdim, 0, m_be.get_stream()>>>(
            x_pred.get_const_buffer(), x_truth.get_const_buffer(),
            y.get_buffer(), sample_size, sample_spatial_size,
            sample_channel_size, thread_work_size);
  }

  if (m_num_procs_per_sample > 1) {
    Al::Allreduce<Al::NCCLBackend, DataType>(
        y.get_buffer(), num_samples,
        Al::ReductionOperator::sum, *m_al.get());
  }

  return 0;
}

template <typename Tensor>
int MeanSquaredErrorCUDNN::backward(const Tensor &x_pred, const Tensor &x_truth,
                                    Tensor &dy, Tensor &dx_pred,
                                    Tensor &dx_truth) {
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
      << "Mean squared error BP: " << dy << ", " << dx_pred << ", " << dx_truth;

  if (m_num_procs_per_sample > 1) {
    const auto num_samples = x_pred.get_local_shape()[-1];
    Al::Bcast<Al::NCCLBackend, DataType>(
        dy.get_buffer(), num_samples, 0,
        *m_al.get());
  }

  constexpr int block_size = 256;
  constexpr int thread_work_size = 8;

  // Assumes no halo for simplicity
  assert_eq(dx_pred.get_local_size(), dx_pred.get_local_real_size());
  assert_eq(dx_truth.get_local_size(), dx_truth.get_local_real_size());

  if (x_pred.get_local_size() == 0) return 0;

  auto num_samples = x_pred.get_local_shape()[-1];
  auto sample_size = x_pred.get_local_size() / num_samples;
  auto num_blocks_per_sample = util::ceil(
      sample_size, (index_t)block_size * thread_work_size);

  dim3 bdim(block_size);
  dim3 gdim(num_blocks_per_sample, num_samples);

  const auto sample_channel_size = x_pred.get_local_shape()[x_pred.get_num_spatial_dims()];
  const auto sample_spatial_size = sample_size / sample_channel_size;
  assert_eq(sample_channel_size*sample_spatial_size, sample_size);

  mean_squared_error::bp_local<DataType, block_size>
      <<<gdim, bdim, 0, m_be.get_stream()>>>(
          x_pred.get_const_buffer(), x_truth.get_const_buffer(),
          dy.get_const_buffer(),
          dx_pred.get_buffer(), dx_truth.get_buffer(),
          sample_size, sample_spatial_size, sample_channel_size,
          thread_work_size);
  return 0;
}

#define PROTO(T)                                                        \
  template int MeanSquaredErrorCUDNN::forward<TensorCUDA<T>>(           \
      const TensorCUDA<T> &x_pred, const TensorCUDA<T> &x_truth,        \
      TensorCUDA<T> &y);                                                \
  template int MeanSquaredErrorCUDNN::backward<TensorCUDA<T>>(          \
      const TensorCUDA<T> &x_pred, const TensorCUDA<T> &x_truth,        \
      TensorCUDA<T> &dy, TensorCUDA<T> &dx_pred,                        \
      TensorCUDA<T> &dx_truth);

PROTO(float)
PROTO(double)
#undef PROTO

} // namespace distconv
