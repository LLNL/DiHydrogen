////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Determine if T is a member of List. */
template <typename T, typename List>
struct MemberVT;

/** @brief Determine if T is a member of List. */
template <typename T, typename List>
constexpr bool MemberV()
{
  return MemberVT<T, List>::value;
}

template <typename T, typename List>
inline constexpr bool Member = MemberV<T, List>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename T>
struct MemberVT<T, Empty> : FalseType
{};

// Match case
template <typename T, typename... Ts>
struct MemberVT<T, TL<T, Ts...>> : TrueType
{};

// Recursive case
template <typename T, typename Head, typename... Tail>
struct MemberVT<T, TL<Head, Tail...>> : MemberVT<T, TL<Tail...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
