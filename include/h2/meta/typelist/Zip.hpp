////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Append.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

// TODO: Generalize to multiple lists.

/** @brief Zip two lists. */
template <typename L1, typename L2>
struct ZipTLT;

/** @brief Zip two lists. */
template <typename L1, typename L2>
using ZipTL = Force<ZipTLT<L1, L2>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <>
struct ZipTLT<Empty, Empty>
{
  using type = Empty;
};

template <typename... List2Ts>
struct ZipTLT<Empty, TL<List2Ts...>>
{
  using type = Empty;
};

template <typename... List1Ts>
struct ZipTLT<TL<List1Ts...>, Empty>
{
  using type = Empty;
};

template <typename T1, typename... List1Ts, typename T2, typename... List2Ts>
struct ZipTLT<TL<T1, List1Ts...>, TL<T2, List2Ts...>>
{
  using type =
    Append<TL<TL<T1, T2>>, Force<ZipTLT<TL<List1Ts...>, TL<List2Ts...>>>>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace tlist
}  // namespace meta
}  // namespace h2
