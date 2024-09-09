#include "distconv/dnn_backend/cross_entropy.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
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

using distconv::tensor::CUDAAllocator;
using distconv::tensor::LocaleMPI;

template <typename DataType>
using TensorCUDA = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;

namespace distconv
{
namespace
{

/*
  - gridDim.y == number of samples
  - Each sample is taken care by gridDim.x blocks
 */
template <typename DataType, int BLOCK_SIZE>
__global__ void fp_local(DataType const* __restrict__ prediction,
                         DataType const* __restrict__ ground_truth,
                         DataType* __restrict__ y,
                         index_t const sample_size,
                         index_t const sample_spatial_size,
                         index_t const sample_channel_size,
                         bool const use_labels,
                         int thread_work_size)
{
  int const tid = threadIdx.x;
  int const sample_idx = blockIdx.y;

  prediction += sample_idx * sample_size;
  ground_truth += sample_idx * sample_size;

  index_t offset = tid + blockIdx.x * BLOCK_SIZE;
  int const offset_stride = BLOCK_SIZE * gridDim.x;
  index_t const offset_limit =
    min(sample_size, offset + offset_stride * thread_work_size);

  auto psum = DataType(0.);
  for (; offset < offset_limit; offset += offset_stride)
  {
    DataType xhat;
    if (use_labels)
    {
      auto const spatial = offset % sample_spatial_size;
      auto const channel = (offset / sample_spatial_size) % sample_channel_size;
      auto const sample = offset / sample_spatial_size / sample_channel_size;
      auto const offset_truth = spatial + sample * sample_spatial_size;
      int const truth_label = ground_truth[offset_truth];
      xhat = DataType(truth_label == channel ? 1. : 0.);
    }
    else
    {
      xhat = ground_truth[offset];
    }
    if (xhat > DataType(0.))
    {
      auto const x = prediction[offset];
      psum += -xhat * log(x);
    }
  }

  using BlockReduce = cubns::BlockReduce<DataType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  psum = BlockReduce(temp_storage).Sum(psum);

  if (tid == 0)
  {
    atomic_add(&y[sample_idx], psum);
  }
}

/*
  - gridDim.y == number of samples
  - Each sample is taken care by gridDim.x blocks
 */
template <typename DataType, int BLOCK_SIZE>
__global__ void bp_local(DataType const* __restrict__ x_pred,
                         DataType const* __restrict__ x_truth,
                         DataType const* __restrict__ dy,
                         DataType* __restrict__ dx_pred,
                         DataType* __restrict__ dx_truth,
                         index_t const sample_size,
                         index_t const sample_spatial_size,
                         index_t const sample_channel_size,
                         bool const use_labels,
                         int thread_work_size)
{
  int const tid = threadIdx.x;
  int const sample_idx = blockIdx.y;

  x_pred += sample_idx * sample_size;
  dx_pred += sample_idx * sample_size;
  x_truth += sample_idx * sample_size;
  dx_truth += sample_idx * sample_size;

  index_t offset = tid + blockIdx.x * BLOCK_SIZE;
  int const offset_stride = BLOCK_SIZE * gridDim.x;
  index_t const offset_limit =
    min(sample_size, offset + offset_stride * thread_work_size);

  auto const dy_sample = dy[sample_idx];
  for (; offset < offset_limit; offset += offset_stride)
  {
    auto const x = x_pred[offset];
    DataType xhat;
    if (use_labels)
    {
      auto const spatial = offset % sample_spatial_size;
      auto const channel = (offset / sample_spatial_size) % sample_channel_size;
      auto const sample = offset / sample_spatial_size / sample_channel_size;
      auto const offset_truth = spatial + sample * sample_spatial_size;
      int const truth_label = x_truth[offset_truth];
      xhat = DataType(truth_label == channel ? 1. : 0.);
    }
    else
    {
      xhat = x_truth[offset];
    }
    dx_pred[offset] =
      (xhat > DataType(0.)) ? -dy_sample * xhat / x : DataType(0.);
    if (!use_labels)
    {
      dx_truth[offset] = -dy_sample * log(x);
    }
  }
}

}  // namespace

template <typename Tensor>
int CrossEntropy<BackendDNNLib>::forward(Tensor const& x_pred,
                                         Tensor const& x_truth,
                                         Tensor& y)
{
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
    << "Cross entropy FP: " << x_pred << ", " << x_truth << ", " << y;

  constexpr int block_size = 256;
  constexpr int thread_work_size = 8;

  // Assumes no halo for simplicity
  assert_eq(x_pred.get_local_size(), x_pred.get_local_real_size());
  assert_eq(x_truth.get_local_size(), x_truth.get_local_real_size());

  auto const num_samples = x_pred.get_local_shape()[-1];

  if (num_samples == 0)
    return 0;

  y.zero(m_stream);

  if (x_pred.get_local_size() > 0)
  {
    auto sample_size = x_pred.get_local_size() / num_samples;
    auto num_blocks_per_sample =
      util::ceil(sample_size, (index_t) block_size * thread_work_size);

    dim3 bdim(block_size);
    dim3 gdim(num_blocks_per_sample, num_samples);

    auto const sample_channel_size =
      x_pred.get_local_shape()[x_pred.get_num_spatial_dims()];
    auto const sample_spatial_size = sample_size / sample_channel_size;
    assert_eq(sample_channel_size * sample_spatial_size, sample_size);

    fp_local<DataType, block_size>
      <<<gdim, bdim, 0, m_stream>>>(x_pred.get_const_buffer(),
                                    x_truth.get_const_buffer(),
                                    y.get_buffer(),
                                    sample_size,
                                    sample_spatial_size,
                                    sample_channel_size,
                                    m_use_labels,
                                    thread_work_size);
  }

  if (m_num_procs_per_sample > 1)
  {
    Al::Allreduce<Al::NCCLBackend, DataType>(
      y.get_buffer(), num_samples, Al::ReductionOperator::sum, *m_al.get());
  }

  return 0;
}

template <typename Tensor>
int CrossEntropy<BackendDNNLib>::backward(Tensor const& x_pred,
                                          Tensor const& x_truth,
                                          Tensor& dy,
                                          Tensor& dx_pred,
                                          Tensor& dx_truth)
{
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
    << "Cross entropy BP: " << dy << ", " << dx_pred << ", " << dx_truth;

  if (m_num_procs_per_sample > 1)
  {
    auto const num_samples = x_pred.get_local_shape()[-1];
    Al::Bcast<Al::NCCLBackend, DataType>(
      dy.get_buffer(), num_samples, 0, *m_al.get());
  }

  constexpr int block_size = 256;
  constexpr int thread_work_size = 8;

  // Assumes no halo for simplicity
  assert_eq(dx_pred.get_local_size(), dx_pred.get_local_real_size());
  assert_eq(dx_truth.get_local_size(), dx_truth.get_local_real_size());

  if (x_pred.get_local_size() == 0)
    return 0;

  auto num_samples = x_pred.get_local_shape()[-1];
  auto sample_size = x_pred.get_local_size() / num_samples;
  auto num_blocks_per_sample =
    util::ceil(sample_size, (index_t) block_size * thread_work_size);

  dim3 bdim(block_size);
  dim3 gdim(num_blocks_per_sample, num_samples);

  auto const sample_channel_size =
    x_pred.get_local_shape()[x_pred.get_num_spatial_dims()];
  auto const sample_spatial_size = sample_size / sample_channel_size;
  assert_eq(sample_channel_size * sample_spatial_size, sample_size);

  bp_local<DataType, block_size>
    <<<gdim, bdim, 0, m_stream>>>(x_pred.get_const_buffer(),
                                  x_truth.get_const_buffer(),
                                  dy.get_const_buffer(),
                                  dx_pred.get_buffer(),
                                  dx_truth.get_buffer(),
                                  sample_size,
                                  sample_spatial_size,
                                  sample_channel_size,
                                  m_use_labels,
                                  thread_work_size);
  return 0;
}

#define PROTO(T)                                                               \
  template int CrossEntropy<BackendDNNLib>::forward<TensorCUDA<T>>(            \
    const TensorCUDA<T>& x_pred,                                               \
    const TensorCUDA<T>& x_truth,                                              \
    TensorCUDA<T>& y);                                                         \
  template int CrossEntropy<BackendDNNLib>::backward<TensorCUDA<T>>(           \
    const TensorCUDA<T>& x_pred,                                               \
    const TensorCUDA<T>& x_truth,                                              \
    TensorCUDA<T>& dy,                                                         \
    TensorCUDA<T>& dx_pred,                                                    \
    TensorCUDA<T>& dx_truth);

PROTO(float)
PROTO(double)
#undef PROTO

}  // namespace distconv
