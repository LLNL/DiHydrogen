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

namespace allreduce_nvshmem {
// TODO: This should be moved to the header file unless device linking
// is enabled.
#ifdef __CUDACC__
// Assume that the root thread has the partial sum. Those held by
// other threads are not used. Only the root thread has the valid
// output value.
template <typename DataType>
__device__ DataType recursive_doubling_block(DataType psum,
                                             size_t num_blocks_per_entry,
                                             int pid, int np, int num_steps,
                                             AllreduceNVSHMEMDevice<DataType> &ar) {

  const int tid = threadIdx.x;
  const int block_offset = blockIdx.x / num_blocks_per_entry;
  const int sync_idx = block_offset * num_steps;
  constexpr SyncType st = SyncType::NONE;

  DataType *tmp_buf = ar.m_buf + (num_blocks_per_entry + num_steps) * block_offset;
  DataType final_sum;

  if (tid == 0) *tmp_buf = psum;

  // TODO: intra-grid partial reduction

  if (tid == 0) {
    // Inter-device reduction
    for (int i = 0; i < num_steps; ++i) {
      int peer = pid ^ (1 << i);
      put_nbi(tmp_buf + 1, tmp_buf, 1, peer);
      ar.m_sync.sync(peer, true, true, st, sync_idx + i);
      tmp_buf[1] += tmp_buf[0];
      ++tmp_buf;
    }
    final_sum = *tmp_buf;
  }

  // TODO: intra-grid scatter
  return final_sum;
}
#endif // __CUDACC__

template <typename DataType>
__global__ void recursive_doubling_block_global(const DataType *send_buf,
                                                DataType *recv_buf,
                                                size_t num_blocks_per_entry,
                                                int pid, int np, int num_steps,
                                                AllreduceNVSHMEMDevice<DataType> ar) {
  const int block_offset = blockIdx.x / num_blocks_per_entry;
  DataType psum;
  if (threadIdx.x == 0) psum = send_buf[block_offset];
  auto sum = recursive_doubling_block(psum, num_blocks_per_entry,
                                      pid, np, num_steps, ar);
  if (threadIdx.x == 0) recv_buf[block_offset] = sum;
}

template <typename T>
struct Vector2;
template <>
struct Vector2<float> {
  using type = float2;
};
template <>
struct Vector2<double> {
  using type = double2;
};
template <>
struct Vector2<int> {
  using type = int2;
};
template <>
struct Vector2<long> {
  using type = long2;
};
} // namespace allreduce_nvshmem

template <typename DataType>
void  AllreduceNVSHMEM<DataType>::recursive_doubling_block(
    const DataType *send_buf, DataType *recv_buf, size_t count) {
  // TODO: For now, use one block per entry
  const int num_blocks_per_entry = 1;
  const int block_size = 32;
  bool vec2 = (count % 2 == 0);
  if (vec2) {
    count /= 2;
  }
  const int grid_size = count * num_blocks_per_entry;

  recursive_doubling_block_setup(count, num_blocks_per_entry);

  auto log_np = std::log2((float)m_np);
  assert_always(std::ceil(log_np) == std::floor(log_np));
  const int num_steps = log_np;

  if (vec2) {
    using OpType = typename allreduce_nvshmem::Vector2<DataType>::type;
    allreduce_nvshmem::recursive_doubling_block_global<OpType><<<
      grid_size, block_size, 0, m_stream>>>(
          (const OpType*)send_buf, (OpType*)recv_buf,
          num_blocks_per_entry, m_pid, m_np, num_steps, get_for_device<OpType>());
  } else {
    allreduce_nvshmem::recursive_doubling_block_global<DataType><<<
      grid_size, block_size, 0, m_stream>>>(
          send_buf, recv_buf, num_blocks_per_entry,
          m_pid, m_np, num_steps, get_for_device());
  }
}

#define DEFINE_RECURSIVE_DOUBLING(TYPE)                                 \
  template                                                              \
  void AllreduceNVSHMEM<TYPE>::recursive_doubling_block(                \
      const TYPE *send_buf, TYPE *recv_buf, size_t count);
DEFINE_RECURSIVE_DOUBLING(float)
DEFINE_RECURSIVE_DOUBLING(double)
DEFINE_RECURSIVE_DOUBLING(int)
DEFINE_RECURSIVE_DOUBLING(long)
#undef DEFINE_RECURSIVE_DOUBLING

} // namespace tensor
} // namespace distconv
