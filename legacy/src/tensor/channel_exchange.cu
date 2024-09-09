#include "distconv/tensor/channel_exchange.hpp"

namespace distconv
{
namespace tensor
{
namespace internal
{

/**
 * @param src Input buffer.
 * @param dst Destination buffer.
 * @param num_samples Number of local samples in src.
 * @param num_channels Number of channels per sample in src.
 * @param num_dests Number of ranks participating in the reduce-scatter.
 * @param src_size Size of the src buffer.
 * @param sample_size Size of a single sample in src.
 * @param channel_size Size of a single channel.
 * @param channels_per_dest Number of channels each destination receives.
 * @param dest_size_per_sample Size of a sample the destination will receive.
 *  This is channel_size*channels_per_dest.
 * @param dst_size Size of the buffer each destination will receive.
 *  This is dest_size_per_sample * num_samples.
 */
template <typename DataType>
__global__ void pack_for_rs_kernel(DataType const* __restrict__ src,
                                   DataType* __restrict__ dst,
                                   size_t const num_samples,
                                   size_t const num_channels,
                                   size_t const num_dests,
                                   size_t const src_size,
                                   size_t const sample_size,
                                   size_t const channel_size,
                                   size_t const channels_per_dest,
                                   size_t const dest_size_per_sample,
                                   size_t const dst_size)
{
  // Input has complete samples.
  // Want to be able to scatter channels of each sample to their destination.
  // This will swizzle the src into a buffer such that the channels to go to
  // each destination are contiguous and ordered correctly.
  size_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t const num_threads = blockDim.x * gridDim.x;
  for (size_t pos = gid; pos < src_size; pos += num_threads)
  {
    size_t const sample = pos / sample_size;
    size_t const channel = (pos - sample * sample_size) / channel_size;
    size_t const dest = channel / channels_per_dest;
    size_t const offset =
      pos - sample * sample_size - dest * dest_size_per_sample;
    size_t const dest_idx =
      dest * dst_size + sample * dest_size_per_sample + offset;
    dst[dest_idx] = src[pos];
  }
}

/**
 * @param packed_buf Packed input buffer.
 * @param dst Destination buffer.
 * @param num_samples Number of local samples.
 * @param num_sources Number of ranks participating in the allgather.
 * @param size Size of the packed_buf buffer.
 * @param source_buf_size Size of buffer each rank contributes to the allgather.
 *  This is num_samples*source_sample_size (or size / num_sources).
 * @param source_sample_size Size of a sample from a source.
 *  This is source_buf_size / num_samples.
 * @param dest_sample_size Size of a sample on the destination.
 */
template <typename DataType>
__global__ void unpack_from_ag_kernel(DataType const* __restrict__ packed_buf,
                                      DataType* __restrict__ dst,
                                      size_t const num_samples,
                                      size_t const num_sources,
                                      size_t const size,
                                      size_t const source_buf_size,
                                      size_t const source_sample_size,
                                      size_t const dest_sample_size)
{
  // Need to reorder the data in input to reassemble samples.
  // The allgather is conducted on the original buffer, which is ordered
  // as <local channels of sample 0> <local channels of sample 1> ...
  // which results in non-contiguous data after gathering.
  // This will swizzle src into a buffer so that the channels of each sample
  // are contiguous.
  size_t const gid = threadIdx.x + blockIdx.x * blockDim.x;
  size_t const num_threads = blockDim.x * gridDim.x;
  for (size_t pos = gid; pos < size; pos += num_threads)
  {
    size_t const source = pos / source_buf_size;
    size_t const sample = (pos - source * source_buf_size) / source_sample_size;
    size_t const offset =
      pos - sample * source_sample_size - source * source_buf_size;
    size_t const dest_idx =
      sample * dest_sample_size + source * source_sample_size + offset;
    dst[dest_idx] = packed_buf[pos];
  }
}

}  // namespace internal

template <>
void ChannelExchange<float>::pack_for_rs(TensorType& src,
                                         TensorType& dst,
                                         float* dst_buf,
                                         size_t comm_size,
                                         h2::gpu::DeviceStream stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  dim3 grid_dim((src.get_local_size() + block_size - 1) / block_size);
  auto src_shape = src.get_local_shape();
  internal::pack_for_rs_kernel<<<grid_dim, block_dim, 0, stream>>>(
    src.get_base_ptr(),
    dst_buf,
    src_shape[-1],
    src_shape[-2],
    comm_size,
    src.get_local_size(),
    get_sample_size(src),
    get_channel_size(src),
    dst.get_local_shape()[-2],
    get_sample_size(dst),
    dst.get_local_size());
}

template <>
void ChannelExchange<double>::pack_for_rs(TensorType& src,
                                          TensorType& dst,
                                          double* dst_buf,
                                          size_t comm_size,
                                          h2::gpu::DeviceStream stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  dim3 grid_dim((src.get_local_size() + block_size - 1) / block_size);
  auto src_shape = src.get_local_shape();
  internal::pack_for_rs_kernel<<<grid_dim, block_dim, 0, stream>>>(
    src.get_base_ptr(),
    dst_buf,
    src_shape[-1],
    src_shape[-2],
    comm_size,
    src.get_local_size(),
    get_sample_size(src),
    get_channel_size(src),
    dst.get_local_shape()[-2],
    get_sample_size(dst),
    dst.get_local_size());
}

template <>
void ChannelExchange<float>::unpack_from_ag(TensorType& src,
                                            TensorType& dst,
                                            float* packed_buf,
                                            size_t comm_size,
                                            h2::gpu::DeviceStream stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  dim3 grid_dim((src.get_local_size() + block_size - 1) / block_size);
  auto src_shape = src.get_local_shape();
  internal::unpack_from_ag_kernel<<<grid_dim, block_dim, 0, stream>>>(
    packed_buf,
    dst.get_base_ptr(),
    src_shape[-1],
    comm_size,
    dst.get_local_size(),
    src.get_local_size(),
    get_sample_size(src),
    get_sample_size(dst));
}

template <>
void ChannelExchange<double>::unpack_from_ag(TensorType& src,
                                             TensorType& dst,
                                             double* packed_buf,
                                             size_t comm_size,
                                             h2::gpu::DeviceStream stream)
{
  constexpr int block_size = 256;
  dim3 block_dim(block_size);
  dim3 grid_dim((dst.get_local_size() + block_size - 1) / block_size);
  auto src_shape = src.get_local_shape();
  internal::unpack_from_ag_kernel<<<grid_dim, block_dim, 0, stream>>>(
    packed_buf,
    dst.get_base_ptr(),
    src_shape[-1],
    comm_size,
    dst.get_local_size(),
    src.get_local_size(),
    get_sample_size(src),
    get_sample_size(dst));
}

}  // namespace tensor
}  // namespace distconv
