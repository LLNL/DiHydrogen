////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Expand.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Return a typelist from applying metafunction to each type.
 *
 *  Alias for ExpandT.
 */
template <template <typename> class F, typename... Ts>
using MapT = ExpandT<F, Ts...>;

/** @brief Return a typelist from applying metafunction to each type.
 *
 *  Alias for Expand.
 */
template <template <typename> class F, typename... Ts>
using Map = Expand<F, Ts...>;

/** @brief Return a typelist from applying metafunction to each type
 *         in the typelist.
 *
 *  Alias for ExpandTL.
 */
template <template <typename> class F, typename List>
using MapTLT = ExpandTLT<F, List>;

/** @brief Return a typelist from applying metafunction to each type
 *         in the typelist.
 *
 *  Alias for ExpandTLT.
 */
template <template <typename> class F, typename List>
using MapTL = ExpandTL<F, List>;

} // namespace tlist
} // namespace meta
} // namespace h2
