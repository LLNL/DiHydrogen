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

/** @brief Create a type list with N copies of T */
template <typename T, unsigned long N>
struct RepeatT;

/** @brief Create a type list with N copies of T */
template <typename T, unsigned long N>
using Repeat = Force<RepeatT<T, N>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <typename T>
struct RepeatT<T, 0UL>
{
    using type = Empty;
};

template <typename T, unsigned long N>
struct RepeatT
{
    using type = Cons<T, Repeat<T, N - 1UL>>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
