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

#include "h2/meta/TypeList.hpp"
#include "h2/utils/Error.hpp"

#include <cstdint>
#include <limits>
#include <type_traits>

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
struct IsH2StorageType
  : std::bool_constant<std::is_trivially_copyable_v<T> && !std::is_pointer_v<T>>
{};

/** Helper variable for `IsH2StorageType`. */
template <typename T>
inline constexpr bool IsH2StorageType_v = IsH2StorageType<T>::value;

/** Assert that `type` is a storage type. */
#define H2_ASSERT_STORAGE_TYPE_ALWAYS(type, ...)                               \
  H2_ASSERT_ALWAYS(IsH2StorageType_v<type>, __VA_ARGS__)
/** Assert that `type` is a storage type only in debug mode. */
#define H2_ASSERT_STORAGE_TYPE_DEBUG(type, ...)                                \
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
struct IsH2ComputeType : std::false_type
{};

// Float types:

template <>
struct IsH2ComputeType<float> : std::true_type
{};
template <>
struct IsH2ComputeType<double> : std::true_type
{};
// TODO: FP16 and BF16 support.

// Integral types:

// Sanity-check for insane platforms:
#if !defined(INT32_MAX) || !defined(UINT32_MAX)
#error "No int32_t or uint32_t, please fix your world"
#endif

template <>
struct IsH2ComputeType<std::int32_t> : std::true_type
{};
template <>
struct IsH2ComputeType<std::uint32_t> : std::true_type
{};

/** Helper variable for `IsH2ComputeType`. */
template <typename T>
inline constexpr bool IsH2ComputeType_v = IsH2ComputeType<T>::value;

/** Assert that `type` is a compute type. */
#define H2_ASSERT_COMPUTE_TYPE_ALWAYS(type, ...)                               \
  H2_ASSERT_ALWAYS(IsH2ComputeType_v<type>, __VA_ARGS__)
/** Assert that `type` is a compute type in debug mode. */
#define H2_ASSERT_COMPUTE_TYPE_DEBUG(type, ...)                                \
  H2_ASSERT_DEBUG(IsH2ComputeType_v<type>, __VA_ARGS__)

// Helpers to explicitly enumerate compute types:

/** List of floating point compute types. */
using FloatComputeTypes = meta::TL<float, double>;

/** List of integral compute types. */
using IntegralComputeTypes = meta::TL<std::int32_t, std::uint32_t>;

/** List of all compute types. */
using ComputeTypes =
  meta::tlist::Append<FloatComputeTypes, IntegralComputeTypes>;

/** Number of compute types. */
constexpr unsigned long NumComputeTypes = meta::tlist::Length<ComputeTypes>;

// Wrap types for runtime dispatch:

/** Manage runtime type information for H2. */
struct TypeInfo
{
  /** Type used to represent tokens for dynamic dispatch. */
  using TokenType = std::uint8_t;
  /** Max value for a token. */
  static constexpr TokenType max_token = std::numeric_limits<TokenType>::max();
  /** Minimum value for user-defined tokens. */
  static constexpr TokenType min_user_token = 32;

  /** Helper to construct TypeInfo with a given token and type. */
  template <typename T>
  static TypeInfo make(TokenType token_)
  {
    static_assert(!std::is_same_v<T, void>,
                  "Cannot construct type info for void");
    return TypeInfo(token_, sizeof(T), &typeid(T));
  }

  /** Get the type token. */
  inline TokenType get_token() const noexcept { return token; }
  /** Get the size of the type. */
  inline std::size_t get_size() const noexcept { return type_size; }
  /** Get the associated type_info. */
  inline const std::type_info* get_type_info() const noexcept
  {
    return type_info;
  }

private:
  TypeInfo(TokenType token_,
           std::size_t type_size_,
           const std::type_info* type_info_)
    : token(token_), type_size(type_size_), type_info(type_info_)
  {
    H2_ASSERT_DEBUG(token_ <= max_token,
                    "Cannot construct type info with a token ",
                    token,
                    " that exceeds the max ",
                    max_token);
  }

  /** Token representing the type for dynamic dispatch. */
  TokenType token;
  /** Size of the type (i.e., `sizeof(T)`). */
  std::size_t type_size;
  /** Pointer to the `std::type_info` associated with the type. */
  const std::type_info* type_info;
};

/** Equality for TypeInfo. */
inline bool operator==(const TypeInfo& t1, const TypeInfo& t2)
{
  return *t1.get_type_info() == *t2.get_type_info();
}

/** Inequality for TypeInfo. */
inline bool operator!=(const TypeInfo& t1, const TypeInfo& t2)
{
  return *t1.get_type_info() != *t2.get_type_info();
}

/** Support printing TypeInfo. */
inline std::ostream& operator<<(std::ostream& os, const TypeInfo& tinfo)
{
  os << "TypeInfo(" << tinfo.get_type_info()->name()
     << ", token=" << tinfo.get_token() << ")";
  return os;
}

/** Get the TypeInfo for a given type. */
template <typename T>
inline TypeInfo get_h2_type()
{
  if constexpr (IsH2ComputeType_v<T>)
  {
    return TypeInfo::make<T>(meta::tlist::Find<ComputeTypes, T>);
  }
  else
  {
    return TypeInfo::make<T>(TypeInfo::max_token);
  }
}

/** True if a type is a native H2 compute type. */
inline bool is_h2_compute_type(const TypeInfo& ti)
{
  return ti.get_token() < NumComputeTypes;
}

/**
 * True if a type is a compute type.
 *
 * This is any type that has a token other than the max token.
 */
inline bool is_compute_type(const TypeInfo& ti)
{
  return ti.get_token() < TypeInfo::max_token;
}

} // namespace h2
