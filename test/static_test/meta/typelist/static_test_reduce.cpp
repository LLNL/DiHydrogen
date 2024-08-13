////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Reduce.hpp"

using namespace h2::meta;

template <typename T, typename U>
struct Add;

template <typename T, T vT, typename U, U vU>
struct Add<std::integral_constant<T, vT>, std::integral_constant<U, vU>>
{
    using type = std::integral_constant<U, vT+vU>;
};

using AccInit = std::integral_constant<int, 0>;
using List = TL<std::integral_constant<int, 1>,
                std::integral_constant<int, 2>,
                std::integral_constant<int, 3>>;
using Result = std::integral_constant<int, 6>;

static_assert(
    EqV<tlist::FoldlTL<Add, AccInit, tlist::Empty>, AccInit>(), "Foldl of nothing returns the initial accumulator.");
static_assert(
    EqV<tlist::FoldlTL<Add, AccInit, List>, Result>(), "Foldl returns the correct result.");

static_assert(
    EqV<tlist::FoldrTL<Add, AccInit, tlist::Empty>, AccInit>(), "Foldl of nothing returns the initial accumulator.");
static_assert(
    EqV<tlist::FoldrTL<Add, AccInit, List>, Result>(), "Foldl returns the correct result.");
