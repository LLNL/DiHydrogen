#include "distconv/cudnn/softmax.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"

#include <limits>

#include <cub/block/block_reduce.cuh>

using distconv::tensor::LocaleMPI;
using distconv::tensor::CUDAAllocator;

template <typename DataType>
using TensorCUDA = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;
using SoftmaxCUDNN = distconv::Softmax<distconv::cudnn::BackendCUDNN>;

namespace distconv {
namespace softmax {

constexpr int block_size = 256;

template <typename DataType>
struct exp;

template <>
struct exp<float> {
  __device__ __forceinline__ float operator()(float x) const {
    return ::expf(x);
  }
};

template <>
struct exp<double> {
  __device__ __forceinline__ double operator()(double x) const {
    return ::exp(x);
  }
};

template <typename DataType>
struct id {
  __device__ __forceinline__ DataType operator()(DataType x) const {
    return x;
  }
};

template <typename DataType>
struct mul {
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) const {
    return x * y;
  }
};

template <typename DataType>
struct div {
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) const {
    return x / y;
  }
};

template <typename DataType>
struct sum {
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) const {
    return x + y;
  }
};

template <typename DataType>
struct max {
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) const {
    return (x >= y) ? x : y;
  }
};

template <typename DataType>
struct atomic_max {
  __device__ __forceinline__ DataType operator()(DataType *addr, DataType value) const;
};

template <>
struct atomic_max<float> {
  __device__ __forceinline__ float operator()(float *addr, float value) const {
    float old;
    old = (value >= 0) ?
        __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));
    return old;
  }
};

template <typename DataType>
struct atomic_add {
  __device__ __forceinline__ DataType operator()(DataType *addr,
                                                 DataType value) const {
    return atomicAdd(addr, value);
  }
};

template <typename DataType>
DataType get_min() {
  return std::sqrt(std::numeric_limits<DataType>::min());
}

template <typename Tensor>
void set_kernel_params(const Tensor &tensor,
                       int &num_samples, size_t &sample_size,
                       dim3 &gdim) {
  int num_blocks = 80; // == V100 #SMs

  num_samples = tensor.get_local_shape()[-1];
  if (num_samples == 0) return;
  sample_size = tensor.get_local_size() / num_samples;
  if (sample_size == 0) return;
  int num_blocks_per_sample = util::ceil(num_blocks, num_samples);
  gdim = dim3(num_blocks_per_sample, num_samples);
}

template <typename DataType, int BLOCK_SIZE,
          typename Map, typename Reduce, typename AtomicReduce>
__global__ void reduce_per_sample_kernel(const DataType * __restrict__ x,
                                         size_t sample_size,
                                         Map map,
                                         Reduce reduce,
                                         AtomicReduce atomic_reduce,
                                         DataType * __restrict__ reduction) {
  auto num_blocks_per_sample = gridDim.x;
  size_t work_per_block = (sample_size + sample_size - 1) / num_blocks_per_sample;
  size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
  size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
  int sample_idx = blockIdx.y;

  x += sample_idx * sample_size;

  DataType local_sum = DataType(0);
  for (; sample_offset < block_end; sample_offset += BLOCK_SIZE) {
    auto x_i = x[sample_offset];
    local_sum = reduce(local_sum, map(x_i));
  }

  using BlockReduce = cub::BlockReduce<DataType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    atomic_reduce(reduction + sample_idx, block_sum);
  }
}

template <typename DataType, int BLOCK_SIZE,
          typename Map, typename Reduce, typename AtomicReduce>
__global__ void reduce_per_sample_kernel(const DataType * __restrict__ x,
                                         const DataType * __restrict__ y,
                                         size_t sample_size,
                                         Map map,
                                         Reduce reduce,
                                         AtomicReduce atomic_reduce,
                                         DataType * __restrict__ reduction) {
  auto num_blocks_per_sample = gridDim.x;
  size_t work_per_block = (sample_size + sample_size - 1) / num_blocks_per_sample;
  size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
  size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
  int sample_idx = blockIdx.y;

  x += sample_idx * sample_size;
  y += sample_idx * sample_size;

  DataType local_sum = DataType(0);
  for (; sample_offset < block_end; sample_offset += BLOCK_SIZE) {
    auto x_i = x[sample_offset];
    auto y_i = y[sample_offset];
    local_sum = reduce(local_sum, map(x_i, y_i));
  }

  using BlockReduce = cub::BlockReduce<DataType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    atomic_reduce(reduction + sample_idx, block_sum);
  }
}

template <typename DataType, int BLOCK_SIZE, typename Map>
__global__ void map_per_sample_kernel(const DataType * __restrict__ x,
                                      const DataType * __restrict__ y,
                                      const DataType * __restrict__ sample_values,
                                      size_t sample_size,
                                      DataType * __restrict__ z,
                                      Map map) {
  auto num_blocks_per_sample = gridDim.x;
  size_t work_per_block = (sample_size + sample_size - 1) / num_blocks_per_sample;
  size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
  size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
  int sample_idx = blockIdx.y;

  x += sample_idx * sample_size;
  y += sample_idx * sample_size;
  z += sample_idx * sample_size;

  const auto sample_value = sample_values[sample_idx];

  for (; sample_offset < block_end; sample_offset += BLOCK_SIZE) {
    auto x_i = x[sample_offset];
    auto y_i = y[sample_offset];
    auto z_i = map(x_i, y_i, sample_value);
    z[sample_offset] = z_i;
  }
}

template <typename DataType, int BLOCK_SIZE,
          typename Map, typename Reduce, typename AtomicReduce>
__global__ void map_and_reduce_per_sample_kernel(
    const DataType * __restrict__ x,
    size_t sample_size,
    const DataType * __restrict__ sample_values,
    Map map,
    Reduce reduce,
    AtomicReduce atomic_reduce,
    DataType * __restrict__ y,
    DataType * __restrict__ reduction) {
  auto num_blocks_per_sample = gridDim.x;
  size_t work_per_block = (sample_size + sample_size - 1) / num_blocks_per_sample;
  size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
  size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
  int sample_idx = blockIdx.y;

  x += sample_idx * sample_size;
  y += sample_idx * sample_size;

  const auto sample_value = sample_values[sample_idx];

  DataType local_sum = DataType(0);
  for (; sample_offset < block_end; sample_offset += BLOCK_SIZE) {
    auto x_i = x[sample_offset];
    x_i = map(x_i, sample_value);
    local_sum = reduce(local_sum, x_i);
    y[sample_offset] = x_i;
  }

  using BlockReduce = cub::BlockReduce<DataType, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto block_sum = BlockReduce(temp_storage).Sum(local_sum);
  if (threadIdx.x == 0) {
    atomic_reduce(reduction + sample_idx, block_sum);
  }
}

template <typename DataType, int BLOCK_SIZE, typename Map>
__global__ void update_per_sample_kernel(
    DataType * __restrict__ x,
    const DataType * __restrict__ sample_values,
    size_t sample_size,
    Map map) {
  auto num_blocks_per_sample = gridDim.x;
  size_t work_per_block = (sample_size + sample_size - 1) / num_blocks_per_sample;
  size_t sample_offset = blockIdx.x * work_per_block + threadIdx.x;
  size_t block_end = min((blockIdx.x + 1) * work_per_block, sample_size);
  int sample_idx = blockIdx.y;

  x += sample_idx * sample_size;

  const auto sample_val = sample_values[sample_idx];

  for (; sample_offset < block_end; sample_offset += BLOCK_SIZE) {
    auto x_i = map(x[sample_offset], sample_val);
    x[sample_offset] = x_i;
  }
}

template <typename Tensor, typename DataType>
void compute_max(const Tensor &tensor, DataType *sample_max,
                 cudaStream_t stream) {
  dim3 gdim;
  int num_samples;
  size_t sample_size;
  set_kernel_params(tensor, num_samples, sample_size, gdim);

  if (num_samples == 0 || sample_size == 0) {
    return;
  }

  reduce_per_sample_kernel
      <DataType, block_size, id<DataType>, max<DataType>,
       atomic_max<DataType>>
      <<<gdim, block_size, 0, stream>>>(
          tensor.get_base_ptr(), sample_size,
          id<DataType>(),
          max<DataType>(),
          atomic_max<DataType>(),
          sample_max);
}

template <typename DataType>
struct exp_shifted {
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) {
    return exp<DataType>()(x - y);
  }
};

template <typename Tensor, typename DataType>
void compute_exp(const Tensor &x, const DataType *sample_max,
                 Tensor &y, DataType *sample_exp,
                 cudaStream_t stream) {
  dim3 gdim;
  int num_samples;
  size_t sample_size;
  set_kernel_params(x, num_samples, sample_size, gdim);

  if (num_samples == 0 || sample_size == 0) {
    return;
  }

  map_and_reduce_per_sample_kernel
      <DataType, block_size,
       exp_shifted<DataType>, sum<DataType>,
       atomic_add<DataType>>
      <<<gdim, block_size, 0, stream>>>(
          x.get_base_ptr(), sample_size, sample_max,
          exp_shifted<DataType>(), sum<DataType>(), atomic_add<DataType>(),
          y.get_base_ptr(), sample_exp);
}

template <typename DataType>
struct SoftmaxOp {
  DataType m_min_output;
  SoftmaxOp(DataType min_output): m_min_output(min_output) {}
  __device__ __forceinline__ DataType operator()(DataType x, DataType y) {
    return ::max(x / y, m_min_output);
  }
};

template <typename Tensor, typename DataType>
void compute_softmax(const DataType *sample_exp, Tensor &output_tensor,
                     cudaStream_t stream) {
  dim3 gdim;
  int num_samples;
  size_t sample_size;
  set_kernel_params(output_tensor, num_samples, sample_size, gdim);

  if (num_samples == 0 || sample_size == 0) {
    return;
  }

  update_per_sample_kernel
      <DataType, block_size, SoftmaxOp<DataType>>
      <<<gdim, block_size, 0, stream>>>(
          output_tensor.get_base_ptr(),
          sample_exp,
          sample_size,
          SoftmaxOp<DataType>(get_min<DataType>()));
}

template <typename Tensor, typename DataType>
void bp_dotproduct(const Tensor &y, const Tensor &dy, DataType *sample_dp,
                   cudaStream_t stream) {
  dim3 gdim;
  int num_samples;
  size_t sample_size;
  set_kernel_params(y, num_samples, sample_size, gdim);

  if (num_samples == 0 || sample_size == 0) {
    return;
  }

  reduce_per_sample_kernel
      <DataType, block_size,
       mul<DataType>, sum<DataType>, atomic_add<DataType>>
      <<<gdim, block_size, 0, stream>>>(
          y.get_base_ptr(), dy.get_base_ptr(), sample_size,
          mul<DataType>(), sum<DataType>(), atomic_add<DataType>(),
          sample_dp);
}

template <typename DataType>
struct bp_compute_func {
  DataType m_min_output;
  bp_compute_func(DataType min_output): m_min_output(min_output) {}

  __device__ __forceinline__ DataType operator()(DataType y, DataType dy,
                                                 DataType dp) {
    if (y > m_min_output) {
      return y * (dy - dp);
    } else {
      return DataType(0);
    }
  }
};

template <typename Tensor, typename DataType>
void bp_compute_gradient(const Tensor &y, const Tensor &dy,
                         DataType *sample_dp, Tensor &dx,
                         cudaStream_t stream) {
  dim3 gdim;
  int num_samples;
  size_t sample_size;
  set_kernel_params(y, num_samples, sample_size, gdim);

  if (num_samples == 0 || sample_size == 0) {
    return;
  }

  map_per_sample_kernel
      <DataType, block_size, bp_compute_func<DataType>>
      <<<gdim, block_size, 0, stream>>>(
          y.get_base_ptr(), dy.get_base_ptr(), sample_dp,
          sample_size, dx.get_base_ptr(),
          bp_compute_func<DataType>(get_min<DataType>()));
}

} // namespace softmax

template <typename Tensor>
int SoftmaxCUDNN::forward(const Tensor &x, Tensor &y) {
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
      << "Softmax FP: " << x << ", " << y;

  auto num_samples = x.get_local_shape()[-1];

  if (num_samples == 0) {
    return 0;
  }

  auto stream = m_be.get_stream();

  auto &mempool = internal::RuntimeCUDA::get_device_memory_pool();
  auto ws_size = num_samples * sizeof(DataType);
  DataType *sample_max = static_cast<DataType*>(
      mempool.get(ws_size, stream));
  DataType *sample_exp = static_cast<DataType*>(
      mempool.get(ws_size, stream));

  DISTCONV_CHECK_CUDA(cudaMemsetAsync(
      sample_max, 0, ws_size, stream));
  DISTCONV_CHECK_CUDA(cudaMemsetAsync(
      sample_exp, 0, ws_size, stream));

  // compute sample-wise max
  softmax::compute_max(x, sample_max, stream);
  allreduce(sample_max, num_samples, true);

  // compute summation of exp
  softmax::compute_exp(x, sample_max, y, sample_exp, stream);
  allreduce(sample_exp, num_samples, false);

  // update the output
  softmax::compute_softmax(sample_exp, y, stream);

  mempool.release(sample_max);
  mempool.release(sample_exp);
  return 0;
}

template <typename Tensor>
int SoftmaxCUDNN::backward(const Tensor &y, const Tensor &dy,
                           Tensor &dx) {
  using DataType = typename Tensor::data_type;
  util::MPIPrintStreamDebug()
      << "Softmax BP: " << y << ", " << dy << ", " << dx;

  auto num_samples = dx.get_local_shape()[-1];

  if (num_samples == 0) {
    return 0;
  }

  auto &mempool = internal::RuntimeCUDA::get_device_memory_pool();
  auto ws_size = num_samples * sizeof(DataType);
  auto stream = m_be.get_stream();
  DataType *sample_dp = static_cast<DataType*>(
      mempool.get(ws_size, stream));

  DISTCONV_CHECK_CUDA(cudaMemsetAsync(
      sample_dp, 0, ws_size, stream));

  // compute sample-wise max
  softmax::bp_dotproduct(y, dy, sample_dp, stream);
  allreduce(sample_dp, num_samples, false);

  softmax::bp_compute_gradient(y, dy, sample_dp, dx, stream);

  mempool.release(sample_dp);
  return 0;
}

template
int SoftmaxCUDNN::forward<TensorCUDA<float>>(const TensorCUDA<float> &x,
                                             TensorCUDA<float> &y);

template
int SoftmaxCUDNN::backward<TensorCUDA<float>>(const TensorCUDA<float> &y,
                                              const TensorCUDA<float> &dy,
                                              TensorCUDA<float> &dx);

} // namespace distconv
