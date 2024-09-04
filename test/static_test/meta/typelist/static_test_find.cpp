////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Find.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

using TList = TL<int, float, bool>;

// Testing Find
static_assert(FindV<TList, int>() == 0UL, "Find int in list");
static_assert(FindV<TList, bool>() == 2UL, "Find bool in list");
static_assert(FindV<TL<int, TList, bool>, TList>() == 1UL, "Find TL in list");
static_assert(
    FindV<Empty, bool>() == static_cast<unsigned long>(-1),
    "Find in an empty list.");
static_assert(
    FindV<TL<int>, bool>() == static_cast<unsigned long>(-1),
    "Find nonexistent type in list.");
