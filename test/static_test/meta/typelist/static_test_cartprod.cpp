////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/CartProd.hpp"

using namespace h2::meta;


using TList1 = TL<char>;
using TList2 = TL<char, short, int>;
using TList3 = TL<float, double, long double>;

// CartProd with an empty list always gives an empty list:
static_assert(EqV<tlist::CartProd<tlist::Empty, tlist::Empty>, tlist::Empty>());
static_assert(EqV<tlist::CartProd<tlist::Empty, TList1>, tlist::Empty>());
static_assert(EqV<tlist::CartProd<tlist::Empty, TList2>, tlist::Empty>());
static_assert(EqV<tlist::CartProd<TList1, tlist::Empty>, tlist::Empty>());
static_assert(EqV<tlist::CartProd<TList2, tlist::Empty>, tlist::Empty>());

// CartProd with non-empty lists:
static_assert(EqV<tlist::CartProd<TList1, TList1>, TL<TL<char, char>>>());
static_assert(EqV<tlist::CartProd<TList1, TList2>,
                  TL<TL<char, char>, TL<char, short>, TL<char, int>>>());
static_assert(EqV<tlist::CartProd<TList2, TList3>,
                  TL<TL<char, float>,
                     TL<char, double>,
                     TL<char, long double>,
                     TL<short, float>,
                     TL<short, double>,
                     TL<short, long double>,
                     TL<int, float>,
                     TL<int, double>,
                     TL<int, long double>>>());
