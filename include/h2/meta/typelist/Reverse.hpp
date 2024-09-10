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
/** @brief Reverse a typelist. */
template <typename List>
struct ReverseT;

/** @brief Reverse a typelist. */
template <typename List>
using Reverse = Force<ReverseT<List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace internal
{
template <typename L, typename RL>
struct ReverseImpl;

template <typename RL>
struct ReverseImpl<Empty, RL>
{
  using type = RL;
};

template <typename T, typename... Ts, typename RL>
struct ReverseImpl<TL<T, Ts...>, RL>
{
  using type = Force<ReverseImpl<TL<Ts...>, Cons<T, RL>>>;
};

}  // namespace internal

template <typename List>
struct ReverseT
{
  using type = Force<internal::ReverseImpl<List, Empty>>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}  // namespace tlist
}  // namespace meta
}  // namespace h2
