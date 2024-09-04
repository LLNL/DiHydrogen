////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/ToFromTuple.hpp"

using namespace h2::meta;

static_assert(
    Eq<tlist::ToTuple<tlist::Empty>, std::tuple<>>,
    "Empty typelist gives empty tuple.");

static_assert(
    Eq<tlist::FromTuple<std::tuple<>>, tlist::Empty>,
    "Empty tuple gives empty typelist.");

static_assert(
    Eq<tlist::ToTuple<TL<char, short, int>>, std::tuple<char, short, int>>,
    "Nontrivial typelist gives nontrivial tuple.");

static_assert(
    Eq<tlist::FromTuple<std::tuple<int, float, double>>, TL<int, float, double>>,
    "Nontrivial tuple gives nontrivial typelist.");
