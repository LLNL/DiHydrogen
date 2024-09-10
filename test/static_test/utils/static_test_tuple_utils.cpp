////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/tuple_utils.hpp"

#include <type_traits>

using namespace h2;

using EmptyTuple = std::tuple<>;
using Tuple1 = std::tuple<int>;
using Tuple2 = std::tuple<int, float>;
using Tuple3 = std::tuple<int, float, char>;

static_assert(std::is_same_v<TupleRemoveFirst_t<Tuple1>, std::tuple<>>);
static_assert(std::is_same_v<TupleRemoveFirst_t<Tuple2>, std::tuple<float>>);
static_assert(
  std::is_same_v<TupleRemoveFirst_t<Tuple3>, std::tuple<float, char>>);

static_assert(std::is_same_v<TupleCat_t<EmptyTuple>, std::tuple<>>);
static_assert(std::is_same_v<TupleCat_t<EmptyTuple, EmptyTuple>, std::tuple<>>);
static_assert(std::is_same_v<TupleCat_t<Tuple1>, std::tuple<int>>);
static_assert(std::is_same_v<TupleCat_t<Tuple1, EmptyTuple>, std::tuple<int>>);
static_assert(std::is_same_v<TupleCat_t<EmptyTuple, Tuple1>, std::tuple<int>>);
static_assert(std::is_same_v<TupleCat_t<Tuple1, Tuple1>, std::tuple<int, int>>);
static_assert(
  std::is_same_v<TupleCat_t<Tuple1, Tuple2>, std::tuple<int, int, float>>);
static_assert(
  std::is_same_v<TupleCat_t<Tuple2, Tuple1>, std::tuple<int, float, int>>);
