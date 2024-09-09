#include <distconv_config.hpp>

#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "distconv/util/util_gpu.hpp"

#include <sstream>
#include <vector>

#if H2_HAS_CUDA
using gpuStream_t = cudaStream_t;
#elif H2_HAS_ROCM
using gpuStream_t = hipStream_t;
#endif

namespace distconv
{

namespace
{
template <int ND>
using Array = tensor::Array<ND>;
using Shape = tensor::Shape;

template <int ND>
__device__ Array<ND> get_idx(size_t linear_offset, Array<ND> const& shape)
{
  Array<ND> idx;
#pragma unroll
  for (int i = 0; i < ND; ++i)
  {
    idx[i] = linear_offset % shape[i];
    linear_offset /= shape[i];
  }
  return idx;
}

/**
   Find the destination rank and the offset in its packing buffer.

   @param src_local_idx local index in the source tensor.
   @param dst_locale_shape the locale shape of the destination tensor.
   @param rank_limits the partitioning limit of each rank.
   @param dst_rank destination rank.
   @param dst_offset offset in the packing buffer.
 */
template <int ND>
__device__ void find_destination(Array<ND> const& src_local_idx,
                                 Array<ND> const& src_local_shape,
                                 Array<ND> const& dst_locale_shape,
                                 int const* __restrict__ rank_limits,
                                 int& dst_rank,
                                 size_t& dst_offset)
{
  dst_rank = 0;
  dst_offset = 0;
  int rank_dim_offset = 1;
  size_t local_linear_offset = 1;
  int rank_limits_idx = 0;
#pragma unroll
  for (int i = 0; i < ND; ++i)
  {
    // Locate the i-th dim index of the destination rank
    int dst_rank_idx;
    int dst_buffer_offset;
    int dst_buffer_dim;
#ifdef DISTCONV_OPTIMIZE_FIND_DESTINATION
    if (rank_limits[rank_limits_idx] == -1)
    {
      auto src_global_index =
        src_local_idx[i] + rank_limits[rank_limits_idx + 1];
      int dst_local_dim = rank_limits[rank_limits_idx + 2];
      dst_rank_idx = src_global_index / dst_local_dim;
      dst_buffer_offset =
        min(src_global_index % dst_local_dim, src_local_idx[i]);
      dst_buffer_dim =
        min((int) (src_local_shape[i] - (src_local_idx[i] - dst_buffer_offset)),
            dst_local_dim);
    }
    else
    {
#endif
      // The if-condition below always holds for some j, so dst_rank_idx
      // is always assigned some value. Initialize just to suppress
      // compiler warnings.
      dst_rank_idx = 0;
      for (int j = 0; j < (int) dst_locale_shape[i]; ++j)
      {
        if (src_local_idx[i] < rank_limits[rank_limits_idx + j])
        {
          dst_rank_idx = j;
          break;
        }
      }
      int dst_rank_base =
        dst_rank_idx == 0 ? 0 : rank_limits[rank_limits_idx + dst_rank_idx - 1];
      dst_buffer_offset = src_local_idx[i] - dst_rank_base;
      dst_buffer_dim =
        rank_limits[rank_limits_idx + dst_rank_idx] - dst_rank_base;
#ifdef DISTCONV_OPTIMIZE_FIND_DESTINATION
    }
#endif

    dst_rank += dst_rank_idx * rank_dim_offset;
    rank_dim_offset *= dst_locale_shape[i];

    dst_offset += dst_buffer_offset * local_linear_offset;
    local_linear_offset *= dst_buffer_dim;

    rank_limits_idx += dst_locale_shape[i];
  }
}

template <int ND>
__device__ size_t get_strided_offset(size_t offset,
                                     Array<ND> const& shape,
                                     Array<ND> const& strides)
{
  size_t real_offset = 0;
#pragma unroll
  for (int i = 0; i < ND; ++i)
  {
    auto x = offset % shape[i];
    real_offset += x * strides[i];
    offset /= shape[i];
  }
  return real_offset;
}

#define PACK_USE_SHMEM
template <int ND, typename DataType, bool packed>
__global__ void pack_kernel(const DataType* src,
                            const Array<ND> src_local_shape,
                            const Array<ND> src_strides,
                            const Array<ND> dst_locale_shape,
                            const int* __restrict__ rank_limits,
                            DataType* __restrict__ buf,
                            const int* __restrict__ displs)
{
  size_t const size = src_local_shape.get_size();
  size_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t const num_threads = blockDim.x * gridDim.x;

#ifdef PACK_USE_SHMEM
  extern __shared__ int shm[];
  int rank_limits_size = 0;
  int displs_size = 1;
#pragma unroll
  for (int i = 0; i < ND; ++i)
  {
    rank_limits_size += dst_locale_shape[i];
    displs_size *= dst_locale_shape[i];
  }
  int* rank_limits_s = shm;
  int* displs_s = &(shm[rank_limits_size]);
  for (int i = threadIdx.x; i < rank_limits_size; i += blockDim.x)
  {
    rank_limits_s[i] = rank_limits[i];
  }
  for (int i = threadIdx.x; i < displs_size; i += blockDim.x)
  {
    displs_s[i] = displs[i];
  }
  __syncthreads();
#endif

  for (size_t offset = gid; offset < size; offset += num_threads)
  {
    size_t src_offset =
      packed ? offset
             : get_strided_offset(offset, src_local_shape, src_strides);
    DataType v = src[src_offset];
    Array<ND> const idx = get_idx(offset, src_local_shape);
    int rank;
    size_t dst_offset;
#ifdef PACK_USE_SHMEM
    find_destination(
      idx, src_local_shape, dst_locale_shape, rank_limits_s, rank, dst_offset);
    buf[displs_s[rank] + dst_offset] = v;
#else
    find_destination(
      idx, src_local_shape, dst_locale_shape, rank_limits, rank, dst_offset);
#if 0
    printf("rank: %d, displs[rank]: %d, dst_offset: %d\n",
           rank, (int)displs[rank], (int)dst_offset);
#endif
    buf[displs[rank] + dst_offset] = v;
#endif
#if 0
    if (offset < 8) {
      printf("pack at %d (strided: %d) : %f, rank: %d, displs: %d, dst_offset: %d\n",
             (int)offset,
             (int)get_strided_offset(offset, src_local_shape, src_strides),
             v, rank, displs[rank], (int)(dst_offset));
    }
#endif
  }
}

template <typename DataType, bool packed>
void pack_kernel_dispatch(DataType const* src,
                          Shape const& src_local_shape,
                          IndexVector const& src_strides,
                          Shape const& dst_locale_shape,
                          int const* rank_limits,
                          DataType* buf,
                          int const* displs,
                          dim3 grid_dim,
                          dim3 block_dim,
                          int shm_size,
                          gpuStream_t stream)
{
  int const num_dims = src_local_shape.num_dims();

#define CALL_KERNEL(ND)                                                        \
  pack_kernel<ND, DataType, packed>                                            \
    <<<grid_dim, block_dim, shm_size, stream>>>(src,                           \
                                                Array<ND>(src_local_shape),    \
                                                Array<ND>(src_strides),        \
                                                Array<ND>(dst_locale_shape),   \
                                                rank_limits,                   \
                                                buf,                           \
                                                displs)

  switch (num_dims)
  {
  case 1: CALL_KERNEL(1); break;
  case 2: CALL_KERNEL(2); break;
  case 3: CALL_KERNEL(3); break;
  case 4: CALL_KERNEL(4); break;
  case 5: CALL_KERNEL(5); break;
  case 6: CALL_KERNEL(6); break;
  default:
    distconv::util::MPIPrintStreamError() << "Unsupported dimension";
    throw std::exception();
  }
#undef CALL_KERNEL
}

template <typename DataType, bool packed>
void pack(DataType const* src,
          Shape const& src_local_shape,
          IndexVector const& src_strides,
          Shape const& dst_locale_shape,
          int const* rank_limits,
          DataType* buf,
          int const* displs,
          gpuStream_t stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  size_t work_size = src_local_shape.get_size();
  dim3 grid_dim((work_size + block_size - 1) / block_size);
#ifdef PACK_USE_SHMEM
  int shm_size = dst_locale_shape.reduce_sum() * sizeof(int)
                 + dst_locale_shape.reduce_prod() * sizeof(int);
#else
  int shm_size = 0;
#endif
  pack_kernel_dispatch<DataType, packed>(src,
                                         src_local_shape,
                                         src_strides,
                                         dst_locale_shape,
                                         rank_limits,
                                         buf,
                                         displs,
                                         grid_dim,
                                         block_dim,
                                         shm_size,
                                         stream);
}

#define PACK_USE_SHMEM
template <int ND, typename DataType, bool packed>
__global__ void unpack_kernel2(DataType* tensor,
                               const Array<ND> local_shape,
                               const Array<ND> strides,
                               const Array<ND> locale_shape,
                               const int* __restrict__ rank_limits,
                               const DataType* __restrict__ packed_buf,
                               const int* __restrict__ displs)
{
  size_t const size = local_shape.get_size();
  size_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t const num_threads = blockDim.x * gridDim.x;

#ifdef PACK_USE_SHMEM
  extern __shared__ int shm[];
  int rank_limits_size = 0;
  int displs_size = 1;
#pragma unroll
  for (int i = 0; i < ND; ++i)
  {
    rank_limits_size += locale_shape[i];
    displs_size *= locale_shape[i];
  }
  int* rank_limits_s = shm;
  int* displs_s = &(shm[rank_limits_size]);
  for (int i = threadIdx.x; i < rank_limits_size; i += blockDim.x)
  {
    rank_limits_s[i] = rank_limits[i];
  }
  for (int i = threadIdx.x; i < displs_size; i += blockDim.x)
  {
    displs_s[i] = displs[i];
  }
  __syncthreads();
#endif

  for (size_t offset = gid; offset < size; offset += num_threads)
  {
    size_t src_offset =
      packed ? offset : get_strided_offset(offset, local_shape, strides);
    Array<ND> const idx = get_idx(offset, local_shape);
    int rank;
    size_t dst_offset;
#ifdef PACK_USE_SHMEM
    find_destination(
      idx, local_shape, locale_shape, rank_limits_s, rank, dst_offset);
    DataType v = packed_buf[displs_s[rank] + dst_offset];
#else
    find_destination(
      idx, local_shape, locale_shape, rank_limits, rank, dst_offset);
    DataType v = packed_buf[displs_s[rank] + dst_offset];
#endif
    tensor[src_offset] = v;
  }
}

template <typename DataType, bool packed>
void unpack_kernel_dispatch(DataType* tensor,
                            Shape const& local_shape,
                            IndexVector const& strides,
                            Shape const& locale_shape,
                            int const* rank_limits,
                            DataType const* packed_buf,
                            int const* displs,
                            dim3 grid_dim,
                            dim3 block_dim,
                            int shm_size,
                            gpuStream_t stream)
{
  int const num_dims = local_shape.num_dims();

#define CALL_KERNEL(ND)                                                        \
  unpack_kernel2<ND, DataType, packed>                                         \
    <<<grid_dim, block_dim, shm_size, stream>>>(tensor,                        \
                                                Array<ND>(local_shape),        \
                                                Array<ND>(strides),            \
                                                Array<ND>(locale_shape),       \
                                                rank_limits,                   \
                                                packed_buf,                    \
                                                displs)

  switch (num_dims)
  {
  case 1: CALL_KERNEL(1); break;
  case 2: CALL_KERNEL(2); break;
  case 3: CALL_KERNEL(3); break;
  case 4: CALL_KERNEL(4); break;
  case 5: CALL_KERNEL(5); break;
  case 6: CALL_KERNEL(6); break;
  default:
    util::MPIPrintStreamError() << "Unsupported dimension";
    throw std::exception();
  }
#undef CALL_KERNEL
}

template <typename DataType, bool packed>
void unpack(DataType* dst,
            Shape const& shape,
            IndexVector const& strides,
            Shape const& locale_shape,
            int const* rank_limits,
            DataType const* buf,
            int const* displs,
            gpuStream_t stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  size_t work_size = shape.get_size();
  dim3 grid_dim((work_size + block_size - 1) / block_size);
#ifdef PACK_USE_SHMEM
  int shm_size = locale_shape.reduce_sum() * sizeof(int)
                 + locale_shape.reduce_prod() * sizeof(int);
#else
  int shm_size = 0;
#endif
  unpack_kernel_dispatch<DataType, packed>(dst,
                                           shape,
                                           strides,
                                           locale_shape,
                                           rank_limits,
                                           buf,
                                           displs,
                                           grid_dim,
                                           block_dim,
                                           shm_size,
                                           stream);
}

}  // namespace

namespace tensor
{

template <typename DataType>
void TensorMPICUDAShuffler<DataType>::shuffle(DataType const* src,
                                              DataType* dst,
                                              gpuStream_t stream,
                                              bool is_forward)
{
  // Poiners can be null if they are empty, which can happen in MPI
  // local tensors
  // assert_always(src != nullptr);
  // assert_always(dst != nullptr);

  int const* rank_limits_fwd = get_rank_limits_fwd(is_forward);
  int const* rank_limits_bwd = get_rank_limits_bwd(is_forward);
  int const* send_counts = get_send_counts(is_forward);
  int const* recv_counts = get_recv_counts(is_forward);
  int const* send_displs_h = get_send_displs_h(is_forward);
  int const* recv_displs_h = get_recv_displs_h(is_forward);
  int const* send_displs_d = get_send_displs_d(is_forward);
  int const* recv_displs_d = get_recv_displs_d(is_forward);

  int const num_ranks = get_src_locale_shape(is_forward).get_size();
  size_t const send_buffer_size =
    get_src_local_shape(is_forward).get_size() * sizeof(DataType);
  DataType* send_buf = get_src_buf(is_forward, stream);
  size_t const recv_buffer_size =
    get_dst_local_shape(is_forward).get_size() * sizeof(DataType);
  DataType* recv_buf = get_dst_buf(is_forward, stream);

  if (send_buffer_size && is_src_split_root(is_forward))
  {
    if (get_src_overlap(is_forward).reduce_sum() == 0)
    {
      pack<DataType, true>(src,
                           get_src_local_shape(is_forward),
                           get_src_strides(is_forward),
                           get_dst_locale_shape(is_forward),
                           rank_limits_fwd,
                           send_buf,
                           send_displs_d,
                           stream);
    }
    else
    {
      pack<DataType, false>(src,
                            get_src_local_shape(is_forward),
                            get_src_strides(is_forward),
                            get_dst_locale_shape(is_forward),
                            rank_limits_fwd,
                            send_buf,
                            send_displs_d,
                            stream);
    }
  }

#if 0
  {
    std::stringstream send_counts_ss;
    std::stringstream recv_counts_ss;
    std::stringstream send_displs_ss;
    std::stringstream recv_displs_ss;
    for (int i = 0; i < num_ranks; ++i) {
      send_counts_ss << " " << send_counts[i];
      recv_counts_ss << " " << recv_counts[i];
      send_displs_ss << " " << send_displs_h[i];
      recv_displs_ss << " " << recv_displs_h[i];
    }
    util::MPIPrintStreamDebug()
        << "Alltoallv: "
        << "send_counts:" << send_counts_ss.str()
        << ", send_displs:" << send_displs_ss.str()
        << ", recv_counts:" << recv_counts_ss.str()
        << ", recv_displs:" << recv_displs_ss.str()
        << "\n";
  }
#endif

  transfer(
    send_buf, send_buffer_size, recv_buf, recv_buffer_size, is_forward, stream);

  // unpack
  if (recv_buffer_size && is_dst_split_root(is_forward))
  {
    if (get_dst_overlap(is_forward).reduce_sum() == 0)
    {
      unpack<DataType, true>(dst,
                             get_dst_local_shape(is_forward),
                             get_dst_strides(is_forward),
                             get_src_locale_shape(is_forward),
                             rank_limits_bwd,
                             recv_buf,
                             recv_displs_d,
                             stream);
    }
    else
    {
      unpack<DataType, false>(dst,
                              get_dst_local_shape(is_forward),
                              get_dst_strides(is_forward),
                              get_src_locale_shape(is_forward),
                              rank_limits_bwd,
                              recv_buf,
                              recv_displs_d,
                              stream);
    }
  }

  release_buf(send_buf);
  release_buf(recv_buf);
}

#define INSTANTIATE_SHUFFLE(TYPE)                                              \
  template <>                                                                  \
  void TensorMPICUDAShuffler<TYPE>::shuffle_forward(                           \
    const TYPE* src, TYPE* dst, gpuStream_t stream)                            \
  {                                                                            \
    shuffle(src, dst, stream, true);                                           \
  };                                                                           \
  template <>                                                                  \
  void TensorMPICUDAShuffler<TYPE>::shuffle_backward(                          \
    const TYPE* src, TYPE* dst, gpuStream_t stream)                            \
  {                                                                            \
    shuffle(src, dst, stream, false);                                          \
  };

INSTANTIATE_SHUFFLE(float)
INSTANTIATE_SHUFFLE(double)
INSTANTIATE_SHUFFLE(int)
INSTANTIATE_SHUFFLE(unsigned)
INSTANTIATE_SHUFFLE(short)
INSTANTIATE_SHUFFLE(unsigned short)
INSTANTIATE_SHUFFLE(long)
INSTANTIATE_SHUFFLE(unsigned long)

}  // namespace tensor
}  // namespace distconv

#ifdef PACK_USE_SHMEM
#undef PACK_USE_SHMEM
#endif
