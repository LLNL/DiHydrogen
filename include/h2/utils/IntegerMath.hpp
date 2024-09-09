////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2_config.hpp"

#include "Error.hpp"
#include "h2/gpu/macros.hpp"
#include "h2/meta/Core.hpp"

#include <cstdint>
#include <stdexcept>
#ifdef H2_HAS_GPU
#include "h2/gpu/runtime.hpp"
#endif

namespace h2
{

// FIXME (trb 03/16/2023): These should continue migrating elsewhere,
// perhaps to the meta directory?
template <typename T>
struct IntegerTraits
{
  using type = T;
  using signed_type = std::make_signed_t<T>;
  using unsigned_type = std::make_unsigned_t<T>;
  static constexpr int nbits = static_cast<int>(sizeof(T) * 8);
};

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

/** @brief Determine a type that will store the given number of bytes. */
template <int Bytes>
struct UTypeForBytes;
template <>
struct UTypeForBytes<1>
{
  using type = std::uint8_t;
};
template <>
struct UTypeForBytes<2>
{
  using type = std::uint16_t;
};
template <>
struct UTypeForBytes<3>
{
  using type = std::uint32_t;
};
template <>
struct UTypeForBytes<4>
{
  using type = std::uint32_t;
};
template <>
struct UTypeForBytes<5>
{
  using type = std::uint64_t;
};
template <>
struct UTypeForBytes<6>
{
  using type = std::uint64_t;
};
template <>
struct UTypeForBytes<7>
{
  using type = std::uint64_t;
};
template <>
struct UTypeForBytes<8>
{
  using type = std::uint64_t;
};

template <typename IType>
using SType = typename IntegerTraits<IType>::signed_type;

template <typename IType>
using UType = typename IntegerTraits<IType>::unsigned_type;

template <typename IType>
inline constexpr auto NBits = IntegerTraits<IType>::nbits;

template <typename IType>
inline constexpr bool IsSigned = meta::EqV<SType<IType>, IType>();

template <typename IType>
inline constexpr bool IsUnsigned = meta::EqV<UType<IType>, IType>();

/** @brief Compute `ceil(log_2(d))`.
 *
 *  Mathematically this doesn't exist for `d=0`. We define this
 *  function to return 0 when `d=0`.
 */
template <typename IType, typename = meta::EnableWhen<IsUnsigned<IType>>>
constexpr H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE auto ceillog2(IType const& d)
{
  int ell = 0;
  for (ell = 0; ell < NBits<IType>; ++ell)
    if ((static_cast<IType>(1) << ell) >= d)
      break;
  return ell;
}

/** @brief Determine if n is a power of 2. */
template <typename IType, typename = meta::EnableWhen<IsUnsigned<IType>>>
constexpr H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE auto ispow2(IType const& d)
{
  return (d & (d - 1)) == 0;
}

/** @brief Determine the minimum number of bytes needed to store bits. */
template <typename IType, typename = meta::EnableWhen<IsUnsigned<IType>>>
constexpr H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE auto
byteceil(IType const& bits)
{
  return (bits / 8) + (bits % 8 > 0);
}

/** @brief Computes the upper 32 bits of `x*y`. */
H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE uint32_t mulhi(uint32_t x,
                                                      uint32_t y) noexcept
{
#if H2_GPU_DEVICE_COMPILING
  return __umulhi(x, y);
#else
  return static_cast<uint32_t>(
    (static_cast<uint64_t>(x) * static_cast<uint64_t>(y)) >> 32);
#endif // H2_GPU_DEVICE_COMPILING
}

/** @brief Computes the upper 64 bits of `x*y`. */
H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE uint64_t mulhi(uint64_t x,
                                                      uint64_t y) noexcept
{
#if H2_GPU_DEVICE_COMPILING
  return __umul64hi(x, y);
#else
  return static_cast<uint64_t>(
    (static_cast<__uint128_t>(x) * static_cast<__uint128_t>(y)) >> 64);
#endif // H2_GPU_DEVICE_COMPILING
}

/** @class FastDiv
 *  @brief Compute division and mods quickly.
 *
 *  The division and mod operations are expensive on pretty much every
 *  processor, but especially on GPUs. There are ways to work around
 *  this. This is a persistent container for the "l" and "m'" values
 *  in Figure 4.1 of [this
 *  paper](https://gmplib.org/~tege/divcnst-pldi94.pdf).
 *
 *  The typical use-case here is breaking down a 1-D index into an N-D
 *  mutli-index. This class can be used to compute the quotients and
 *  remainders needed in that calculation. In this case, the constant
 *  divisor that gets stored is the extent of some dimension. Objects
 *  of this class implicitly convert back to their IType so that
 *  extent value does not need to be multiply stored.
 *
 *  The setup is all done on the host for performance reasons.
 *
 *  @todo Is it worth branching when d=2^l? Shame C++ doesn't have
 *        dependent typing...
 */
template <typename IType, typename = meta::EnableWhen<IsUnsigned<IType>>>
class FastDiv
{
  static_assert(IsUnsigned<IType>, "FastDiv for unsigned division only");

public:
  using UInt = UType<IType>;

public:
  FastDiv() : FastDiv(1u) {}
  FastDiv(UInt d) : div_{d}
  {
    H2_ASSERT(d > 0, std::runtime_error, "divisor must be positive.");
    using BigUInt =
      meta::IfThenElse<meta::EqV<UInt, uint32_t>(), uint64_t, __uint128_t>;
    static constexpr auto N = NBits<UInt>;
    static constexpr auto one = static_cast<UInt>(1);
    static constexpr auto bigone = static_cast<BigUInt>(1);
    int ell = ceillog2(d);
    mprime_ =
      static_cast<UInt>(((bigone << N) * ((bigone << ell) - d) / d) + one);
    sh1_ = (ell < 1 ? ell : 1);
    sh2_ = (ell == 0 ? 0 : ell - 1);
  }

  // This lets it masquerade as a dim if needed
  H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE operator UInt const&() const noexcept
  {
    return div_;
  }

  H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE void div(UInt const& in,
                                                  UInt& q) const noexcept
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
  }

  H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE UInt div(UInt const& in) const noexcept
  {
    UInt q;
    div(in, q);
    return q;
  }

  H2_GPU_FORCE_INLINE H2_GPU_HOST_DEVICE void
  divmod(UInt const& in, UInt& q, UInt& r) const noexcept
  {
    div(in, q);
    r = in - (q * div_);
  }

private:
  UInt div_;
  UInt mprime_;
  int sh1_;
  int sh2_;
}; // class FastDiv
} // namespace h2
