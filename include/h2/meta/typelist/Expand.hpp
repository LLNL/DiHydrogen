////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_EXPAND_HPP_
#define H2_META_TYPELIST_EXPAND_HPP_

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

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <typename> class UnaryT, typename... Ts>
struct ExpandT
{
    using type = TL<UnaryT<Ts>...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_EXPAND_HPP_
