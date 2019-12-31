#include "distconv/tensor/allreduce_nvshmem.hpp"

namespace distconv {
namespace tensor {

namespace {
template <typename DataType>
__global__ void copy_kernel(const DataType *src, DataType *dst, size_t count) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t num_threads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < count; i += num_threads) {
    dst[i] = src[i];
  }
}

template <typename DataType>
__global__ void reduce_kernel(const DataType *src, DataType *dst, size_t count) {
  size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t num_threads = blockDim.x * gridDim.x;
  for (size_t i = tid; i < count; i += num_threads) {
    dst[i] += src[i];
  }
}

} // namespace

#define DEFINE_COPY(TYPE)                                               \
  template <>                                                           \
  void AllreduceNVSHMEM<TYPE>::copy(const TYPE *src, TYPE *dst,         \
                                    size_t count) {                     \
    if (src == dst) return;                                             \
    int block_dim = 256;                                                \
    int grid_dim = std::min(128ul, (count + block_dim - 1)/ block_dim); \
    copy_kernel<<<grid_dim, block_dim, 0, m_stream>>>(src, dst, count); \
  }
DEFINE_COPY(float)
DEFINE_COPY(double)
DEFINE_COPY(int)
DEFINE_COPY(long)
#undef DEFINE_COPY

#define DEFINE_REDUCE(TYPE)                                             \
  template <>                                                           \
  void AllreduceNVSHMEM<TYPE>::reduce(const TYPE *src, TYPE *dst,       \
                                      size_t count) {                   \
    int block_dim = 256;                                                \
    int grid_dim = std::min(128ul, (count + block_dim - 1)/ block_dim); \
    reduce_kernel<<<grid_dim, block_dim, 0, m_stream>>>(src, dst, count); \
  }
DEFINE_REDUCE(float)
DEFINE_REDUCE(double)
DEFINE_REDUCE(int)
DEFINE_REDUCE(long)
#undef DEFINE_REDUCE

} // namespace tensor
} // namespace distconv
