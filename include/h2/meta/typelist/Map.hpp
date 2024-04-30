////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_MAP_HPP_
#define H2_META_TYPELIST_MAP_HPP_

#include "Expand.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Alias for Expand. */
template <template <typename> class F, typename... Ts>
using Map = Expand<F, Ts...>;

/** @brief Alias for ExpandTL. */
template <template <typename> class F, typename List>
using MapTL = ExpandTL<F, List>;

} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_MAP_HPP_
