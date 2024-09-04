////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Replace all instances of Old with New in List.
 *  @tparam List The list in which to do replacement.
 *  @tparam Old The type to be replaced.
 *  @tparam New The replacement type.
 */
template <typename List, typename Old, typename New>
struct ReplaceAllT;

/** @brief Replace all instances of Old with New in List. */
template <typename List, typename Old, typename New>
using ReplaceAll = Force<ReplaceAllT<List, Old, New>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename Old, typename New>
struct ReplaceAllT<Empty, Old, New>
{
    using type = Empty;
};

// Ret case
template <typename Old, typename New, typename... Ts>
struct ReplaceAllT<TypeList<Old, Ts...>, Old, New>
    : ConsT<New, ReplaceAll<TypeList<Ts...>, Old, New>>
{};

// Recursive case
template <typename List, typename Old, typename New>
struct ReplaceAllT : ConsT<Car<List>, ReplaceAll<Cdr<List>, Old, New>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
