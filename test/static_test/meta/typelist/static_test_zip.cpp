////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/zip.hpp"

using namespace h2::meta;


using TList1 = TL<char>;
using TList2 = TL<char, short, int>;
using TList3 = TL<float, double, long double>;

// Zip with an empty list always gives an empty list:
static_assert(EqV<tlist::Zip<tlist::Empty, tlist::Empty>, tlist::Empty>());
static_assert(EqV<tlist::Zip<tlist::Empty, TList1>, tlist::Empty>());
static_assert(EqV<tlist::Zip<tlist::Empty, TList2>, tlist::Empty>());
static_assert(EqV<tlist::Zip<TList1, tlist::Empty>, tlist::Empty>());
static_assert(EqV<tlist::Zip<TList2, tlist::Empty>, tlist::Empty>());

// Zip with non-empty, equal-length lists:
static_assert(EqV<tlist::Zip<TList1, TList1>, TL<TL<char, char>>>());
static_assert(EqV<tlist::Zip<TList2, TList2>,
                  TL<TL<char, char>, TL<short, short>, TL<int, int>>>());
static_assert(
    EqV<tlist::Zip<TList2, TList3>,
        TL<TL<char, float>, TL<short, double>, TL<int, long double>>>());

// Zip with unequal-length lists:
static_assert(EqV<tlist::Zip<TList1, TList3>, TL<TL<char, float>>>());
