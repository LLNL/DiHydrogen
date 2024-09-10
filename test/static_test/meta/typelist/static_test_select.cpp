////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Select.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

template <typename T>
using IsInt = EqVT<T, int>;

using TList1 = TL<float, double, int, char, bool>;
using TList2 = TL<unsigned, long, short>;

static_assert(EqV<Select<TList1, IsInt>, int>(),
              "Select an int from the typelist containing an int.");
static_assert(EqV<Select<TList2, IsInt>, Nil>(),
              "Try selecting an int from a typelist with no int.");
