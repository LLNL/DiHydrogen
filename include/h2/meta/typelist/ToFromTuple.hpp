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

/** @brief Produce a tuple from a TypeList */
template <typename List>
struct ToTupleT;

/** @brief Produce a tuple from a TypeList */
template <typename List>
using ToTuple = Force<ToTupleT<List>>;

/** @brief Produce a TypeList from a tuple */
template <typename Tup>
struct FromTupleT;

/** @brief Produce a TypeList from a tuple */
template <typename Tup>
using FromTuple = Force<FromTupleT<Tup>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename... Ts>
struct ToTupleT<TL<Ts...>>
{
    using type = std::tuple<Ts...>;
};

template <typename... Ts>
struct FromTupleT<std::tuple<Ts...>>
{
    using type = TL<Ts...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
