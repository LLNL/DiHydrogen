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

#include <type_traits>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "h2/gpu/macros.hpp"


namespace h2
{

/**
 * Identify a vector type for T with the given vector width.
 *
 * Falls back to T if none is available.
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

template <typename T, std::size_t vec_width>
using VectorType_t = typename VectorTypeForT<T, vec_width>::type;

/**
 * Aligned data suitable for vectorization over vec_width elements.
 */
template <typename T, std::size_t vec_width>
struct alignas(sizeof(T) * vec_width) aligned_data
{
  T data[vec_width];
};

/** Load a vector of data from the given offset. */
template <std::size_t vec_width, typename T>
H2_GPU_DEVICE VectorType_t<T, vec_width> load_vector(const T* data,
                                                     std::size_t offset)
{
  auto vec_data = reinterpret_cast<const VectorType_t<T, vec_width>*>(data);
  return vec_data[offset];
}

/**
 * Return the maximum vector width for ptr, based on its alignment.
 *
 * @note Currently this only considers vector widths of 1, 2, or 4.
 */
template <typename T>
H2_GPU_HOST_DEVICE std::size_t max_vectorization_amount(const T* ptr)
{
  const std::uintptr_t addr = reinterpret_cast<std::uintptr_t>(ptr);
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

}  // namespace h2
