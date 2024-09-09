////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Replace.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

// Testing Replace
static_assert(EqV<Replace<Empty, int, float>, Empty>(),
              "Replacing in an empty list has no effect.");
static_assert(
  EqV<Replace<TL<int, int, int>, int, float>, TL<float, int, int>>(),
  "Replace list of the same type.");

using TList = TL<int, float, double, int, float, double>;
static_assert(
  EqV<Replace<TList, double, char>, TL<int, float, char, int, float, double>>(),
  "Replace in an interesting list.");

static_assert(EqV<Replace<TList, unsigned, char>, TList>(),
              "Replace a nonexistent type has no effect.");
