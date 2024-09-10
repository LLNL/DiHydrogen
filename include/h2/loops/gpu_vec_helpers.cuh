////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

// This file is meant to be included only in source files.
#if !defined(__CUDACC__) && !defined(__HIPCC__)
#error "This file is to only be included in GPU code"
#endif

#pragma once

/** @file
 *
 * Helper utilities for GPU vectorization.
 */

#include <h2_config.hpp>

#include "h2/gpu/macros.hpp"
#include "h2/gpu/runtime.hpp"
#include "h2/utils/const_for.hpp"
#include "h2/meta/TypeList.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace h2
{
namespace gpu
{

/**
 * Identify a vector type for T with the given vector width.
 *
 * Falls back to T if none is available.
 *
 * @tparam T Type to identify a vector version for.
 * @tparam vec_width Width of the desired vector type.
 */
template <typename T, std::size_t vec_width>
struct VectorTypeForT;
template <typename T>
struct VectorTypeForT<T, 1>
{
  using type = T;
};
template <>
struct VectorTypeForT<char, 2>
{
  using type = char2;
};
template <>
struct VectorTypeForT<char, 4>
{
  using type = char4;
};
template <>
struct VectorTypeForT<unsigned char, 2>
{
  using type = uchar2;
};
template <>
struct VectorTypeForT<unsigned char, 4>
{
  using type = uchar4;
};
template <>
struct VectorTypeForT<short, 2>
{
  using type = short2;
};
template <>
struct VectorTypeForT<short, 4>
{
  using type = short4;
};
template <>
struct VectorTypeForT<unsigned short, 2>
{
  using type = ushort2;
};
template <>
struct VectorTypeForT<unsigned short, 4>
{
  using type = ushort4;
};
template <>
struct VectorTypeForT<int, 2>
{
  using type = int2;
};
template <>
struct VectorTypeForT<int, 4>
{
  using type = int4;
};
template <>
struct VectorTypeForT<unsigned int, 2>
{
  using type = uint2;
};
template <>
struct VectorTypeForT<unsigned int, 4>
{
  using type = uint4;
};
template <>
struct VectorTypeForT<long, 2>
{
  using type = long2;
};
template <>
struct VectorTypeForT<long, 4>
{
  using type = long4;
};
template <>
struct VectorTypeForT<unsigned long, 2>
{
  using type = ulong2;
};
template <>
struct VectorTypeForT<unsigned long, 4>
{
  using type = ulong4;
};
template <>
struct VectorTypeForT<long long, 2>
{
  using type = longlong2;
};
template <>
struct VectorTypeForT<long long, 4>
{
  using type = longlong4;
};
template <>
struct VectorTypeForT<unsigned long long, 2>
{
  using type = ulonglong2;
};
template <>
struct VectorTypeForT<unsigned long long, 4>
{
  using type = ulonglong4;
};
template <>
struct VectorTypeForT<float, 2>
{
  using type = float2;
};
template <>
struct VectorTypeForT<float, 4>
{
  using type = float4;
};
template <>
struct VectorTypeForT<double, 2>
{
  using type = double2;
};
template <>
struct VectorTypeForT<double, 4>
{
  using type = double4;
};

template <typename T>
struct VectorTypeForT<T const, 2>
{
  using type = typename VectorTypeForT<T, 2>::type const;
};
template <typename T>
struct VectorTypeForT<T const, 4>
{
  using type = typename VectorTypeForT<T, 4>::type const;
};

template <typename T, std::size_t vec_width>
using VectorType_t = meta::Force<
  VectorTypeForT<std::remove_pointer_t<std::remove_reference_t<T>>, vec_width>>;

/**
 * Access the i'th index of a vector type.
 *
 * If T is not a vector type (width 1), this simply returns the input.
 *
 * @tparam i Index (0-based) of the vector type to return.
 * @tparam vec_width Width of the vector type.
 * @tparam T Underlying type for the vector.
 */
template <std::size_t i, std::size_t vec_width, typename T>
H2_GPU_HOST_DEVICE T& index_vector(VectorType_t<T, vec_width>& vec)
{
  static_assert(i < vec_width, "Invalid vector index");
  static_assert(vec_width <= 4, "Invalid vector width");

  if constexpr (vec_width == 1)
  {
    return vec;
  }
  else
  {
    if constexpr (i == 0)
    {
      return vec.x;
    }
    else if constexpr (i == 1)
    {
      return vec.y;
    }
    else if constexpr (i == 2)
    {
      return vec.z;
    }
    else if constexpr (i == 3)
    {
      return vec.w;
    }
  }
}

/**
 * Obtain a tuple type that contains vector versions of each type in a
 * given type list.
 *
 * @tparam vec_width Width of the desired vector types.
 * @tparam List TypeList of types to be converted to vectors.
 */
template <std::size_t vec_width, typename List>
struct VectorTupleTypeT;
template <std::size_t vec_width, typename... Ts>
struct VectorTupleTypeT<vec_width, meta::TypeList<Ts...>>
{
  using type = std::tuple<VectorType_t<Ts, vec_width>...>;
};
template <std::size_t vec_width, typename List>
using VectorTupleType_t = meta::Force<VectorTupleTypeT<vec_width, List>>;

/**
 * Helper for loading arguments from a tuple of vectorized data into a
 * non-vectorized tuple suitable for calling functions on.
 *
 * @tparam i Index (0-based) of the vector to load.
 * @tparam vec_width Width of the vector type.
 * @tparam List TypeList of the underlying types in the tuple.
 */
template <std::size_t i, std::size_t vec_width, typename List>
struct LoadVectorTuple;
template <std::size_t i, std::size_t vec_width, typename... Ts>
struct LoadVectorTuple<i, vec_width, meta::TypeList<Ts...>>
{
  /**
   * Return a tuple of non-vectorized arguments from a vectorized
   * tuple argument.
   */
  static H2_GPU_HOST_DEVICE std::tuple<Ts...>
  load(VectorTupleType_t<vec_width, meta::TypeList<Ts...>>& vec_tuple)
  {
    std::tuple<Ts...> ret;
    const_for<std::size_t{0}, sizeof...(Ts), std::size_t{1}>([&](auto arg_i) {
      using T = meta::tlist::At<meta::TypeList<Ts...>, arg_i>;
      std::get<arg_i>(ret) =
        index_vector<i, vec_width, T>(std::get<arg_i>(vec_tuple));
    });
    return ret;
  }

  /**
   * Like `load`, but prepend an immediate argument.
   */
  template <typename ImmediateT>
  static H2_GPU_HOST_DEVICE std::tuple<ImmediateT, Ts...> load_with_immediate(
    ImmediateT& imm, VectorTupleType_t<vec_width, meta::TypeList<Ts...>>& vec_tuple)
  {
    std::tuple<ImmediateT, Ts...> ret;
    std::get<0>(ret) = imm;
    const_for<std::size_t{0}, sizeof...(Ts), std::size_t{1}>([&](auto arg_i) {
      using T = meta::tlist::At<meta::TypeList<Ts...>, arg_i>;
      std::get<arg_i + 1>(ret) =
        index_vector<i, vec_width, T>(std::get<arg_i>(vec_tuple));
    });
    return ret;
  }
};

/**
 * Return the maximum vector width for ptr, based on its alignment.
 *
 * @note Currently this only considers vector widths of 1, 2, or 4.
 */
template <typename T>
H2_GPU_HOST_DEVICE std::size_t max_vectorization_amount(T const* ptr)
{
  std::uintptr_t const addr = reinterpret_cast<std::uintptr_t>(ptr);
  if ((addr % std::alignment_of_v<VectorType_t<T, 4>>) == 0)
  {
    return 4;
  }
  else if ((addr % std::alignment_of_v<VectorType_t<T, 2>>) == 0)
  {
    return 2;
  }
  return 1;
}

}  // namespace gpu
}  // namespace h2
