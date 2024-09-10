////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/partial_functions/Apply.hpp"

using namespace h2::meta;
using namespace h2::meta::pfunctions;

namespace
{
template <typename A, typename B, typename C>
struct F;
}  // namespace

static_assert(EqV<Apply<F<int, int, int>, tlist::Empty>, F<int, int, int>>(),
              "Apply returns input when no placeholders present.");

static_assert(EqV<Apply<F<_1, _2, _3>, TL<int>>, F<int, _1, _2>>(),
              "Replacing the first argument.");
static_assert(EqV<Apply<F<_1, _2, _3>, TL<int, char>>, F<int, char, _1>>(),
              "Replacing the first two arguments.");
static_assert(
  EqV<Apply<F<_1, _2, _3>, TL<int, char, bool>>, F<int, char, bool>>(),
  "Replacing all three arguments.");

static_assert(EqV<Apply<F<_1, _3, _2>, TL<int>>, F<int, _2, _1>>(),
              "Replacing the first argument.");
static_assert(EqV<Apply<F<_1, _3, _2>, TL<int, char>>, F<int, _1, char>>(),
              "Replacing the first two arguments.");
static_assert(
  EqV<Apply<F<_1, _3, _2>, TL<int, bool, char>>, F<int, char, bool>>(),
  "Replacing all three arguments.");

static_assert(EqV<Apply<F<_2, _1, _3>, TL<int>>, F<_1, int, _2>>(),
              "Replacing the first argument.");
static_assert(EqV<Apply<F<_2, _1, _3>, TL<int, char>>, F<char, int, _1>>(),
              "Replacing the first two arguments.");
static_assert(
  EqV<Apply<F<_2, _1, _3>, TL<int, char, bool>>, F<char, int, bool>>(),
  "Replacing all three arguments.");

static_assert(EqV<Apply<F<_2, _3, _1>, TL<int>>, F<_1, _2, int>>(),
              "Replacing the first argument.");
static_assert(EqV<Apply<F<_2, _3, _1>, TL<int, char>>, F<char, _1, int>>(),
              "Replacing the first two arguments.");
static_assert(
  EqV<Apply<F<_2, _3, _1>, TL<int, char, bool>>, F<char, bool, int>>(),
  "Replacing all three arguments.");

static_assert(EqV<Apply<F<_3, _1, _2>, TL<int>>, F<_2, int, _1>>(),
              "Replacing the last argument only.");
static_assert(EqV<Apply<F<_3, _1, _2>, TL<int, char>>, F<_1, int, char>>(),
              "Replacing the last two arguments.");
static_assert(
  EqV<Apply<F<_3, _1, _2>, TL<int, char, bool>>, F<bool, int, char>>(),
  "Replacing all arguments backwards.");

static_assert(EqV<Apply<F<_3, _2, _1>, TL<int>>, F<_2, _1, int>>(),
              "Replacing the last argument only.");
static_assert(EqV<Apply<F<_3, _2, _1>, TL<int, char>>, F<_1, char, int>>(),
              "Replacing the last two arguments.");
static_assert(
  EqV<Apply<F<_3, _2, _1>, TL<int, char, bool>>, F<bool, char, int>>(),
  "Replacing all arguments backwards.");
