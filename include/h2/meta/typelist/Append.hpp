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
/** @brief Join multiple lists into one. */
template <typename... Lists>
struct AppendT;

/** @brief Join multiple lists into one */
template <typename... Lists>
using Append = Force<AppendT<Lists...>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Single list
template <typename... ListTs>
struct AppendT<TL<ListTs...>>
{
    using type = TL<ListTs...>;
};

// Two lists
template <typename... ListOneTs, typename... ListTwoTs>
struct AppendT<TypeList<ListOneTs...>, TypeList<ListTwoTs...>>
{
    using type = TL<ListOneTs..., ListTwoTs...>;
};

// Many lists
template <typename FirstList, typename... OtherLists>
struct AppendT<FirstList, OtherLists...>
    : AppendT<FirstList, Append<OtherLists...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
