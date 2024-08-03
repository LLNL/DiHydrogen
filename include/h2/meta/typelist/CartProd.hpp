////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "Append.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

// TODO: Generalize to multiple lists.

/** @brief Construct the Cartesian product of two lists. */
template <typename L1, typename L2>
struct CartProdT;

/** @brief Construct the Cartesian product of two lists. */
template <typename L1, typename L2>
using CartProd = Force<CartProdT<L1, L2>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Case with one empty list:
template <typename... List2Ts>
struct CartProdT<Empty, TL<List2Ts...>>
{
  using type = Empty;
};

// Two lists:
template <typename T, typename... List1Ts, typename... List2Ts>
struct CartProdT<TL<T, List1Ts...>, TL<List2Ts...>>
{
  using type = Append<TL<TL<T, List2Ts>...>,
                      Force<CartProdT<TL<List1Ts...>, TL<List2Ts...>>>>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace tlist
}  // namespace meta
}  // namespace h2
