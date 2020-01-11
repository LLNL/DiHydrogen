#include "distconv/tensor/allreduce_nvshmem.hpp"

using namespace distconv::util::nvshmem;

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

namespace allreduce_nvshmem {

template <typename DataType>
__device__ __forceinline__ void copy_block(DataType *x, const DataType *y, size_t count) {
  if (x == y) return;
  size_t bsize = blockDim.x;
  for (size_t idx = threadIdx.x; idx < count; idx += bsize) {
    x[idx] = y[idx];
  }
}

template <typename DataType>
__device__ __forceinline__ void reduce_block(DataType *x, const DataType *y, size_t count) {
  size_t bsize = blockDim.x;
  for (size_t idx = threadIdx.x; idx < count; idx += bsize) {
    x[idx] += y[idx];
  }
}

template <typename DataType>
__device__ __forceinline__ void swap(DataType *&x, DataType *&y) {
  auto z = x;
  x = y;
  y = z;
}

template <typename DataType>
__global__ void recursive_doubling_kernel(const DataType *send_buf,
                                          DataType *recv_buf,
                                          DataType *tmp_buf,
                                          size_t count, size_t work_per_block,
                                          int pid, int np, int num_steps,
                                          SyncArrayDevice sync) {
  const int tid = threadIdx.x;
  int sync_idx = blockIdx.x * num_steps;
  send_buf += blockIdx.x * work_per_block;
  recv_buf += blockIdx.x * work_per_block;
  tmp_buf += blockIdx.x * work_per_block;
  work_per_block = min(work_per_block, count - blockIdx.x * work_per_block);

  copy_block(recv_buf, send_buf, work_per_block);

  constexpr SyncType st = SyncType::NONE;

  for (int i = 0; i < num_steps; ++i) {
    int peer = pid ^ (1 << i);
    if (i != 0) {
      __syncthreads();
      if (tid == 0) {
        sync.sync(peer, true, true, st, sync_idx + i);
      }
    }
    __syncthreads();
    put_nbi_block(tmp_buf, recv_buf, work_per_block, peer);
    __syncthreads();
    if (tid == 0) {
      sync.sync(peer, true, true, st, sync_idx + i);
    }
    __syncthreads();
    reduce_block(tmp_buf, recv_buf, work_per_block);
    swap(tmp_buf, recv_buf);
  }

  if (num_steps % 2 != 0) {
    copy_block(tmp_buf, recv_buf, work_per_block);
  }
}

} // namespace allreduce_nvshmem

template <typename DataType>
void AllreduceNVSHMEM<DataType>::recursive_doubling(
    const DataType *send_buf, DataType *recv_buf, size_t count) {
  size_t work_per_block;
  int block_size;
  int grid_size;
  set_blocking_params(count, work_per_block, block_size, grid_size);

  auto log_np = std::log2((float)m_np);
  assert_always(std::ceil(log_np) == std::floor(log_np));

  ensure_buffer(count);

  const int num_steps = log_np;

  // Need to have different sync objects for different thread blocks
  m_sync.ensure_size(num_steps * grid_size);

  allreduce_nvshmem::recursive_doubling_kernel<DataType><<<
    grid_size, block_size, 0, m_stream>>>(
        send_buf, recv_buf, static_cast<DataType*>(m_buf.get()),
        count, work_per_block,
        m_pid, m_np, num_steps, m_sync.get_for_device());
}

#define DEFINE_RECURSIVE_DOUBLING(TYPE)                                 \
  template                                                              \
  void AllreduceNVSHMEM<TYPE>::recursive_doubling(                      \
      const TYPE *send_buf, TYPE *recv_buf, size_t count);
DEFINE_RECURSIVE_DOUBLING(float)
DEFINE_RECURSIVE_DOUBLING(double)
DEFINE_RECURSIVE_DOUBLING(int)
DEFINE_RECURSIVE_DOUBLING(long)
#undef DEFINE_RECURSIVE_DOUBLING

namespace allreduce_nvshmem {
template <typename DataType>
__global__ void recursive_doubling_buffered(const DataType *send_buf,
                                            DataType *recv_buf,
                                            DataType *tmp_buf,
                                            size_t count,
                                            size_t work_per_block,
                                            int pid, int np, int num_steps,
                                            SyncArrayDevice sync) {
  const int tid = threadIdx.x;
  int sync_idx = blockIdx.x * num_steps;
  send_buf += blockIdx.x * work_per_block;
  recv_buf += blockIdx.x * work_per_block;
  tmp_buf += blockIdx.x * work_per_block * num_steps;
  work_per_block = min(work_per_block, count - blockIdx.x * work_per_block);

  copy_block(recv_buf, send_buf, work_per_block);

  constexpr SyncType st = SyncType::NONE;

  DataType *par_sum = recv_buf;
  for (int i = 0; i < num_steps; ++i) {
    int peer = pid ^ (1 << i);
    __syncthreads();
    put_nbi_block(tmp_buf, par_sum, work_per_block, peer);
    __syncthreads();
    if (tid == 0) {
      sync.sync(peer, true, true, st, sync_idx + i);
    }
    __syncthreads();
    reduce_block(tmp_buf, par_sum, work_per_block);
    par_sum = tmp_buf;
    tmp_buf += work_per_block;
  }
  copy_block(recv_buf, par_sum, work_per_block);
}
} // namespace allreduce_nvshmem

template <typename DataType>
void AllreduceNVSHMEM<DataType>::recursive_doubling_buffered(
    const DataType *send_buf, DataType *recv_buf, size_t count) {
  size_t work_per_block;
  int block_size;
  int grid_size;
  set_blocking_params(count, work_per_block, block_size, grid_size);

  auto log_np = std::log2((float)m_np);
  assert_always(std::ceil(log_np) == std::floor(log_np));
  const int num_steps = log_np;

  ensure_buffer(count * num_steps);

  // Need to have different sync objects for different thread blocks
  m_sync.ensure_size(num_steps * grid_size);

  allreduce_nvshmem::recursive_doubling_buffered<DataType><<<
    grid_size, block_size, 0, m_stream>>>(
        send_buf, recv_buf, static_cast<DataType*>(m_buf.get()),
        count, work_per_block,
        m_pid, m_np, num_steps, m_sync.get_for_device());
}

#define DEFINE_RECURSIVE_DOUBLING(TYPE)                                 \
  template                                                              \
  void AllreduceNVSHMEM<TYPE>::recursive_doubling_buffered(             \
      const TYPE *send_buf, TYPE *recv_buf, size_t count);
DEFINE_RECURSIVE_DOUBLING(float)
DEFINE_RECURSIVE_DOUBLING(double)
DEFINE_RECURSIVE_DOUBLING(int)
DEFINE_RECURSIVE_DOUBLING(long)
#undef DEFINE_RECURSIVE_DOUBLING

} // namespace tensor
} // namespace distconv
