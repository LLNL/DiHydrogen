////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Append.hpp"
#include "h2/meta/typelist/Length.hpp"

using namespace h2::meta;
using namespace h2::meta::tlist;

// Testing Length
static_assert(LengthV<Empty>() == 0UL, "Length of empty list is zero.");
static_assert(
    LengthV<TypeList<int>>() == 1UL, "Length of TypeList<int> is one.");
static_assert(
    LengthV<TypeList<int, float>>() == 2UL,
    "Length of TypeList<int, float> is two.");
static_assert(
    LengthV<TypeList<int, float, char>>() == 3UL,
    "Length of TypeList<int, float, char> is three.");

// Not sure why I did this... Must have been bored. Oh well...
static_assert(
    LengthV<Append<
            Empty,
            TypeList<char, short, int>,
            Empty,
            TypeList<unsigned char, unsigned short, unsigned int>,
            Empty,
            TypeList<float, double, long double>,
            Empty>>()
        == 9UL,
    "Length of TypeList created by Append.");
