////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/SelectAll.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

template <typename T>
using IsInt = EqVT<T, int>;

using TList1 = TL<float, int, double, int, char, bool>;
using TList2 = TL<unsigned, long, short>;

static_assert(EqV<SelectAll<TList1, IsInt>, TL<int, int>>(),
              "Select all ints from the typelist containing ints.");
static_assert(EqV<SelectAll<TList2, IsInt>, Empty>(),
              "Try selecting all ints from typelist with no int.");
