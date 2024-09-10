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
/** @brief Remove the first instance of a type from a typelist. */
template <typename List, typename T>
struct RemoveT;

/** @brief Remove the first instance of a type from a typelist. */
template <typename List, typename T>
using Remove = Force<RemoveT<List, T>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T>
struct RemoveT<Empty, T>
{
  using type = Empty;
};

// Match case
template <typename T, typename... Ts>
struct RemoveT<TypeList<T, Ts...>, T>
{
  using type = TypeList<Ts...>;
};

// Recursive call
template <typename S, typename... Ts, typename T>
struct RemoveT<TypeList<S, Ts...>, T> : ConsT<S, Remove<TypeList<Ts...>, T>>
{};

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}  // namespace tlist
}  // namespace meta
}  // namespace h2
