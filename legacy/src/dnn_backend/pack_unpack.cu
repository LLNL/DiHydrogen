#include <hip/hip_runtime.h>

namespace
{
template <typename T>
struct IntegerTraits;

template <>
struct IntegerTraits<int32_t>
{
    using type = int32_t;
    using signed_type = int32_t;
    using unsigned_type = uint32_t;
    static constexpr int nbits = 32;
};

template <>
struct IntegerTraits<uint32_t>
{
    using type = uint32_t;
    using signed_type = int32_t;
    using unsigned_type = uint32_t;
    static constexpr int nbits = 32;
};

template <>
struct IntegerTraits<int64_t>
{
    using type = int64_t;
    using signed_type = int64_t;
    using unsigned_type = uint64_t;
    static constexpr int nbits = 64;
};

template <>
struct IntegerTraits<uint64_t>
{
    using type = uint64_t;
    using signed_type = int64_t;
    using unsigned_type = uint64_t;
    static constexpr int nbits = 64;
};

template <typename IType>
using SType = typename IntegerTraits<IType>::signed_type;

template <typename IType>
using UType = typename IntegerTraits<IType>::unsigned_type;

template <typename IType>
inline constexpr auto NBits = IntegerTraits<IType>::nbits;

template <typename IType>
inline constexpr bool IsSigned = std::is_same_v<SType<IType>, IType>;

template <typename IType>
inline constexpr bool IsUnsigned = std::is_same_v<UType<IType>, IType>;

template <typename IType>
__forceinline__ __host__ __device__ auto ceillog2(IType const& d)
{
    static_assert(IsUnsigned<IType>, "divisor should be unsigned");
    static constexpr auto nbits = NBits<IType>;
    int ell = 0;
    for (ell = 0; ell < nbits; ++ell)
        if ((1U << ell) >= d)
            break;
    return ell;
}

__forceinline__ __device__ uint32_t mulhi(uint32_t x, uint32_t y)
{
    return __umulhi(x, y);
}

__forceinline__ __device__ uint64_t mulhi(uint64_t x, uint64_t y)
{
    return __umul64hi(x, y);
}

// Handy container for the "l" and "m'" values in Figure 4.1 of
// https://gmplib.org/~tege/divcnst-pldi94.pdf
template <typename IType>
class FastDiv
{
    static_assert(IsUnsigned<IType>, "FastDiv for unsigned division only");

public:
    using UInt = UType<IType>;

public:
    FastDiv() : FastDiv(1u) {}
    FastDiv(UInt d) : div_{d}
    {
        int ell = ceillog2(d);
        mprime_ =
            static_cast<UInt>(((1ul << 32) * ((1ul << ell) - d) / d) + 1ul);
        sh1_ = min(ell, 1);
        sh2_ = max(ell - 1, 0);
    }

    // This lets it masquerade as a dim if needed
    __host__ __device__ operator UInt const&() const noexcept { return div_; }
    __forceinline__ __host__ __device__ void
    divmod(UInt const& in, UInt& q, UInt& r) const noexcept
    {
        UInt const t1 = mulhi(mprime_, in);
        // There's a warning in the paper not to compute it this way
        // since the sum may overflow N bits. In preliminary tests,
        // overflow was not observed, but the measurable impact on
        // performance was negligible. So safety first and all that...
        // But I'm leaving it here in case anyone wants to reevaluate
        // that claim later on. One shift is better than two. (An
        // alternative approach could be to use 2*N bits for the
        // result of (t1+in) and cast the result of the shift back to
        // N bits before return. I have not looked into any
        // performance implications of this.)
        // q = (t1 + in) >> ell_;
        q = (t1 + ((in - t1) >> sh1_)) >> sh2_;
        r = in - (q * div_);
    }

private:
    UInt div_;
    UInt mprime_;
    int sh1_;
    int sh2_;
};

template <typename T, size_t ND>
struct array
{
    T val[ND];
};

template <typename IType, size_t ND>
__forceinline__ __host__ __device__ auto
get_size(array<FastDiv<IType>, ND> const& dims)
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
             array<FastDiv<IType>, ND> const& dims,
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
                            array<FastDiv<IType>, ND> const dims,
                            array<IType, ND> const src_strides,
                            array<IType, ND> const tgt_strides,
                            float const* src_data,
                            float* tgt_data)
{
    using UInt = UType<IType>;
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
                   hipStream_t stream)
{
    using int_type = uint32_t;
    using FastDivT = FastDiv<int_type>;
    array<int_type, ND> src_strides, tgt_strides;
    array<FastDivT, ND> dims;
    for (size_t i = ND; i > 0; --i)
        dims.val[ND - i] = FastDivT(dims_in[i - 1]);
    std::reverse_copy(src_strides_in, src_strides_in + ND, src_strides.val);
    std::reverse_copy(tgt_strides_in, tgt_strides_in + ND, tgt_strides.val);

    size_t const N = get_size(dims);
    size_t const blk_size = 256; // 64;

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
}

} // namespace

namespace distconv
{
namespace miopen
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
                          hipStream_t stream)
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

} // namespace miopen
} // namespace distconv
