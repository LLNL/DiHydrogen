////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Expand.hpp"
#include "h2/meta/typelist/Map.hpp"

using namespace h2::meta;

template <typename T>
struct X;

using ResultTList1 = TL<X<char>>;
using ResultTList2 = TL<X<char>, X<int>, X<float>>;

static_assert(
    EqV<tlist::Expand<X>, tlist::Nil>(), "Expand no typelist returns nil.");
static_assert(
    EqV<tlist::Expand<X, char>, ResultTList1>(), "Expand single type.");
static_assert(
    EqV<tlist::Expand<X, char, int, float>, ResultTList2>(),
    "Expand over multiple types.");

// "Map" alias
static_assert(
    EqV<tlist::Map<X>, tlist::Nil>(), "Map no typelist returns nil.");
static_assert(
    EqV<tlist::Map<X, char>, ResultTList1>(), "Map single type.");
static_assert(
    EqV<tlist::Map<X, char, int, float>, ResultTList2>(),
    "Map over multiple types.");
