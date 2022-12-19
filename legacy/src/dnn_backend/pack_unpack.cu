#include <hip/hip_runtime.h>

namespace
{
template <typename T, size_t ND>
struct array
{
    T val[ND];
};

template <size_t ND>
__global__
void copy_kernel(float const alpha,
                 float const beta,
                 array<int, ND> const dims,
                 array<int, ND> const src_strides,
                 array<int, ND> const tgt_strides,
                 array<int, ND> const packed_strides,
                 float const* src_data,
                 float* tgt_data)
{
    size_t const max_index = ((size_t) dims.val[0]) * ((size_t) packed_strides.val[0]);
    for (size_t entry1d = threadIdx.x + blockIdx.x * blockDim.x;
         entry1d < max_index;
         entry1d += blockDim.x * gridDim.x)
    {
        size_t entry1d_tmp = entry1d;
        size_t src_idx_1d = 0, tgt_idx_1d = 0;
#pragma unroll
        for (size_t i = 0; i < ND; ++i)
        {
            int const idx_i = (entry1d / packed_strides.val[i]);
            src_idx_1d += idx_i * src_strides.val[i];
            tgt_idx_1d += idx_i * tgt_strides.val[i];
            entry1d = (entry1d % packed_strides.val[i]);
        }
        tgt_data[tgt_idx_1d] = alpha * src_data[src_idx_1d] + beta * tgt_data[tgt_idx_1d];
    }
}

template <size_t ND>
void launch_kernel(float const& alpha,
                   float const& beta,
                   int const* dims_in,
                   int const* src_strides_in,
                   int const* tgt_strides_in,
                   int const* packed_strides_in,
                   float const* src_data,
                   float* tgt_data,
                   hipStream_t stream)
{
  array<int, ND> dims, src_strides, tgt_strides, packed_strides;
  std::copy_n(dims_in, ND, dims.val);
  std::copy_n(src_strides_in, ND, src_strides.val);
  std::copy_n(tgt_strides_in, ND, tgt_strides.val);
  std::copy_n(packed_strides_in, ND, packed_strides.val);

  size_t const N = ((size_t) dims_in[0]) * ((size_t) packed_strides_in[0]);
  size_t const blk_size = 256;

  hipLaunchKernelGGL(copy_kernel,
                     dim3((N+blk_size-1)/blk_size),
                     dim3(blk_size),
                     0,
                     stream,
                     alpha,
                     beta,
                     dims,
                     src_strides,
                     tgt_strides,
                     packed_strides,
                     src_data,
                     tgt_data);
}

}

namespace distconv
{
namespace miopen
{
  // A copy with different strides
void do_gpu_tensor_repack(
    float const& alpha,
    float const& beta,
    size_t const ndims,
    int const* dims,
    int const* src_strides,
    int const* tgt_strides,
    int const* packed_strides,
    float const* src_data,
    float* tgt_data,
    hipStream_t stream)
{
    switch (ndims)
    {
    case 1:
        launch_kernel<1>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 2:
        launch_kernel<2>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 3:
        launch_kernel<3>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 4:
        launch_kernel<4>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 5:
        launch_kernel<5>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 6:
        launch_kernel<6>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
        break;
    case 7:
        launch_kernel<7>(alpha, beta, dims, src_strides, tgt_strides, packed_strides, src_data, tgt_data, stream);
    default:
        throw std::runtime_error("Unsupported ndims.");
    }
}
} // namespace miopen
} // namespace distconv
