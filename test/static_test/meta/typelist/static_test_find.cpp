////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/core/Eq.hpp"
#include "h2/meta/typelist/Find.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

using TList = TL<char, signed char, short, int, long, long long>;

// Testing Find
static_assert(Find<Empty, int> == InvalidIdx, "Find in an empty list.");

static_assert(Find<TList, char> == 0UL, "Find char in list.");
static_assert(Find<TList, int> == 3UL, "Find int in list.");
static_assert(FindV<TList, int>() == 3UL,
              "Find int in list, function version.");
static_assert(Find<TList, long long> == 5UL, "Find long long in list.");

static_assert(Find<TList, float> == InvalidIdx,
              "Find nonexistent type in list.");

// Find returns the first match's index
static_assert(Find<TL<int, double, char, double>, double> == 1UL,
              "Find returns first matching index.");

// Testing FindIf
template <typename T>
using IsInt = EqVT<int, T>;

template <typename T>
using IsDbl = EqVT<double, T>;

static_assert(FindIf<IsInt, TList> == 3UL, "Find first int in list.");
static_assert(FindIf<IsDbl, TList> == InvalidIdx,
              "FindIf returns InvalidIdx if no matching type found.");
