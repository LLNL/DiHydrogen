////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Core H2 types.
 */

#include <h2_config.hpp>

#include <type_traits>

#include <cstdint>

#include "h2/utils/Error.hpp"
#include "h2/meta/TypeList.hpp"


namespace h2
{

// H2 storage types:

/**
 * A type trait with member `value` which will be true if `T` is a
 * supported H2 storage type, and false otherwise.
 *
 * A storage type is one that H2 is able to store (e.g., in a `Tensor`)
 * and copy. A storage type is not necessarily supported by any compute
 * kernels.
 *
 * Currently the only requirement is that `T` be trivially copyable and
 * not a pointer type.
 */
template <typename T>
struct IsH2StorageType : std::bool_constant<std::is_trivially_copyable_v<T>
                                            && !std::is_pointer_v<T>>
{};

/** Helper variable for `IsH2StorageType`. */
template <typename T>
inline constexpr bool IsH2StorageType_v = IsH2StorageType<T>::value;

/** Assert that `type` is a storage type. */
#define H2_ASSERT_STORAGE_TYPE_ALWAYS(type, ...)                \
  H2_ASSERT_ALWAYS(IsH2StorageType_v<type>, __VA_ARGS__)
/** Assert that `type` is a storage type only in debug mode. */
#define H2_ASSERT_STORAGE_TYPE_DEBUG(type, ...)         \
  H2_ASSERT_DEBUG(IsH2StorageType_v<type>, __VA_ARGS__)

// H2 compute types:

/**
 * A type trait with member `value` which will be true if `T` is a
 * supported H2 compute type, and false otherwise.
 *
 * A compute type is an H2 storage type which H2 also generates compute
 * kernels for.
 *
 * This is restricted to an explicit list of types.
 */
template <typename T>
struct IsH2ComputeType : std::false_type {};

// Float types:

template <> struct IsH2ComputeType<float> : std::true_type {};
template <> struct IsH2ComputeType<double> : std::true_type {};
// TODO: FP16 and BF16 support.

// Integral types:

// Sanity-check for insane platforms:
#if !defined(INT32_MAX) || !defined(UINT32_MAX)
#error "No int32_t or uint32_t, please fix your world"
#endif

template <> struct IsH2ComputeType<std::int32_t> : std::true_type {};
template <> struct IsH2ComputeType<std::uint32_t> : std::true_type {};

/** Helper variable for `IsH2ComputeType`. */
template <typename T>
inline constexpr bool IsH2ComputeType_v = IsH2ComputeType<T>::value;

/** Assert that `type` is a compute type. */
#define H2_ASSERT_COMPUTE_TYPE_ALWAYS(type, ...)                \
  H2_ASSERT_ALWAYS(IsH2ComputeType_v<type>, __VA_ARGS__)
/** Assert that `type` is a compute type in debug mode. */
#define H2_ASSERT_COMPUTE_TYPE_DEBUG(type, ...)         \
  H2_ASSERT_DEBUG(IsH2ComputeType_v<type>, __VA_ARGS__)

// Helpers to explicitly enumerate compute types:

/** List of floating point compute types. */
using FloatComputeTypes = meta::TL<float, double>;

/** List of integral compute types. */
using IntegralComputeTypes = meta::TL<std::int32_t, std::uint32_t>;

/** List of all compute types. */
using ComputeTypes =
  meta::tlist::Append<FloatComputeTypes, IntegralComputeTypes>;

}  // namespace h2
