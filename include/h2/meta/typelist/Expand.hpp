////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Expand a template and parameters into a typelist */
template <template <typename> class UnaryT, typename... Ts>
struct ExpandT;

/** @brief Expand a template and parameters into a typelist */
template <template <typename> class UnaryT, typename... Ts>
using Expand = Force<ExpandT<UnaryT, Ts...>>;

/** @brief Expand a template and parameters stored in a typelist into
 *  a typelist.
 */
template <template <typename> class UnaryT, typename TList>
struct ExpandTLT;

/** @brief Expand a template and parameters stored in a typelist into
 *  a typelist.
 */
template <template <typename> class UnaryT, typename TList>
using ExpandTL = Force<ExpandTLT<UnaryT, TList>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <typename> class UnaryT, typename... Ts>
struct ExpandT
{
    using type = TL<UnaryT<Ts>...>;
};

template <template <typename> class UnaryT, typename... Ts>
struct ExpandTLT<UnaryT, TL<Ts...>>
{
    using type = Expand<UnaryT, Ts...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
