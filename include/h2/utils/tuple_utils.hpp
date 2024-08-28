////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Utilities for working with `std::tuple`s.
 */

#include <tuple>


namespace h2
{

/**
 * Defines a type that is a tuple of all types except the first type in
 * the tuple.
 *
 * I.e., this is cdr.
 */
template <typename Tuple>
struct TupleRemoveFirstT;

template <typename First, typename... Rest>
struct TupleRemoveFirstT<std::tuple<First, Rest...>>
{
  using type = std::tuple<Rest...>;
};

template <typename Tuple>
using TupleRemoveFirst_t = typename TupleRemoveFirstT<Tuple>::type;

/**
 * Defines a type that is a tuple containing the concatenation of the
 * types in the given tuples.
 *
 * Essentially, this is a shorthand for `decltype(std::tuple_cat(...))`.
 */
template <typename... Tuples>
struct TupleCatT;

template <typename... Ts>
struct TupleCatT<std::tuple<Ts...>>
{
  using type = std::tuple<Ts...>;
};

template <typename... T1s, typename... T2s>
struct TupleCatT<std::tuple<T1s...>, std::tuple<T2s...>>
{
  using type = std::tuple<T1s..., T2s...>;
};

template <typename Tuple, typename... OtherTuples>
struct TupleCatT<Tuple, OtherTuples...>
    : TupleCatT<Tuple, TupleCatT<OtherTuples...>>
{};

template <typename... Tuples>
using TupleCat_t = typename TupleCatT<Tuples...>::type;

}
