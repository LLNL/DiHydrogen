////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Append.hpp"
#include "Map.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Flatten all typelists into a single typelist */
template <typename... Ts>
struct FlattenT;

/** @brief Flatten all typelists into a single typelist */
template <typename... Ts>
using Flatten = Force<FlattenT<Ts...>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T>
struct Flatten_implT;

template <typename T>
using Flatten_impl = Force<Flatten_implT<T>>;

// Atoms don't need flattening
template <typename T>
struct Flatten_implT
{
    using type = TL<T>;
};

template <>
struct Flatten_implT<Empty>
{
    using type = Empty;
};

// Typelists get extracted and recursively flattened
template <typename... Ts>
struct Flatten_implT<TL<Ts...>>
{
    using type = Flatten<Ts...>;
};

template <typename... Ts>
struct FlattenT
{
    using type = Append<Flatten_impl<Ts>...>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
