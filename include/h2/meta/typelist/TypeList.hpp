////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
/** @struct TypeList
 *  @brief A basic type list.
 *
 *  Functions that act on typelists are in the tlist namespace. There
 *  are basic accessors that offer either Lisp- or Haskell-like
 *  semantics. In a post-C++11 world, Haskell semantics are probably
 *  closer to what is natural in template metaprogramming.
 *
 *  When Lisp-family semantic choices need to be made (e.g., what
 *  happens when you take the car of the empty list), the ANSI Common
 *  Lisp standard is followed.
 *
 *  When ML-family semantic choices need to be made, Haskell
 *  conventions are adopted.
 */
template <typename... Ts>
struct TypeList;

/** @brief A short-hand alias for TypeLists. */
template <typename... Ts>
using TL = TypeList<Ts...>;

/** @brief Basic metamethods on TypeLists. */
namespace tlist
{
/** @brief The empty list. */
using Empty = TypeList<>;

/** @brief The empty list. */
using Nil = Empty;

}  // namespace tlist

// Implementation

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// This gives typelists boolean value semantics. It's not clear if
// this matters.
template <typename... Ts>
struct TypeList : TrueType
{};

template <>
struct TypeList<> : FalseType
{};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace meta
}  // namespace h2
