#include "h2/utils/IntegerMath.hpp"

#if H2_HAS_ROCM
#include <hip/hip_runtime.h>
using Stream_t = hipStream_t;
#elif H2_HAS_CUDA
#include <cuda_runtime.h>
using Stream_t = cudaStream_t;
#endif

namespace {

template <typename T, size_t ND>
struct array
{
    T val[ND];
};

template <typename IType, size_t ND>
__forceinline__ __host__ __device__ auto
get_size(array<h2::FastDiv<IType>, ND> const& dims)
{
    IType size = 1;
#pragma unroll
    for (size_t i = 0; i < ND; ++i)
        size *= dims.val[i];
    return size;
}

template <typename IType, size_t ND>
__forceinline__ __host__ __device__ IType
get_real_idx(IType idx,
             array<h2::FastDiv<IType>, ND> const& dims,
             array<IType, ND> const& strides)
{
    IType real_idx = 0;
    IType q, r;
#pragma unroll
    for (size_t i = 0; i < ND; ++i)
    {
        dims.val[i].divmod(idx, q, r);
        real_idx += r * strides.val[i];
        idx = q;
    }
    return real_idx;
}

template <typename IType, size_t ND>
__global__ void copy_kernel(float const alpha,
                            float const beta,
                            size_t const max_index,
                            array<h2::FastDiv<IType>, ND> const dims,
                            array<IType, ND> const src_strides,
                            array<IType, ND> const tgt_strides,
                            float const* src_data,
                            float* tgt_data)
{
    using UInt = h2::UType<IType>;
    UInt const num_thds = blockDim.x * gridDim.x;

    for (UInt entry1d = threadIdx.x + blockIdx.x * blockDim.x;
         entry1d < max_index;
         entry1d += num_thds)
    {
        UInt src_idx_1d = get_real_idx(entry1d, dims, src_strides);
        UInt tgt_idx_1d = get_real_idx(entry1d, dims, tgt_strides);
        tgt_data[tgt_idx_1d] = src_data[src_idx_1d];
    }
}

template <size_t ND>
void launch_kernel(float const& alpha,
                   float const& beta,
                   int const* dims_in,
                   int const* src_strides_in,
                   int const* tgt_strides_in,
                   float const* src_data,
                   float* tgt_data,
                   Stream_t stream)
{
    using int_type = uint32_t;
    using FastDivT = h2::FastDiv<int_type>;
    array<int_type, ND> src_strides, tgt_strides;
    array<FastDivT, ND> dims;
    for (size_t i = ND; i > 0; --i)
        dims.val[ND - i] = FastDivT(dims_in[i - 1]);
    std::reverse_copy(src_strides_in, src_strides_in + ND, src_strides.val);
    std::reverse_copy(tgt_strides_in, tgt_strides_in + ND, tgt_strides.val);

    size_t const N = get_size(dims);
    size_t const blk_size = 256; // 64;

#if H2_HAS_ROCM
    hipLaunchKernelGGL(copy_kernel,
                       dim3((N + blk_size - 1) / blk_size),
                       dim3(blk_size),
                       0,
                       stream,
                       alpha,
                       beta,
                       N,
                       dims,
                       src_strides,
                       tgt_strides,
                       src_data,
                       tgt_data);
#elif H2_HAS_CUDA
    copy_kernel<<<dim3((N + blk_size - 1) / blk_size),
                  dim3(blk_size),
                  0,
                  stream>>>(
                      alpha,
                      beta,
                      N,
                      dims,
                      src_strides,
                      tgt_strides,
                      src_data,
                      tgt_data);
#endif
}

} // namespace

namespace distconv
{

// A copy with different strides
void do_gpu_tensor_repack(float const& alpha,
                          float const& beta,
                          size_t const ndims,
                          int const* dims,
                          int const* src_strides,
                          int const* tgt_strides,
                          float const* src_data,
                          float* tgt_data,
                          Stream_t stream)
{
    switch (ndims)
    {
    case 1:
        launch_kernel<1>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 2:
        launch_kernel<2>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 3:
        launch_kernel<3>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 4:
        launch_kernel<4>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 5:
        launch_kernel<5>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 6:
        launch_kernel<6>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    case 7:
        launch_kernel<7>(alpha,
                         beta,
                         dims,
                         src_strides,
                         tgt_strides,
                         src_data,
                         tgt_data,
                         stream);
        break;
    default: throw std::runtime_error("Unsupported ndims.");
    }
}

} // namespace distconv
