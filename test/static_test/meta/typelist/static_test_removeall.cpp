////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/RemoveAll.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

using TList = TL<int, float, double, int, float, double>;

// Testing Remove
static_assert(EqV<RemoveAll<TList, int>, TL<float, double, float, double>>(),
              "Remove first entry.");
static_assert(EqV<RemoveAll<TList, float>, TL<int, double, int, double>>(),
              "Remove second entry.");
static_assert(EqV<RemoveAll<TList, double>, TL<int, float, int, float>>(),
              "Remove third entry.");
static_assert(EqV<RemoveAll<TList, bool>, TList>(), "Remove nonexistent type.");
static_assert(EqV<RemoveAll<Empty, bool>, Empty>(),
              "Remove nonexistent type from empty list.");
