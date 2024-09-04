////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Sort.hpp"

using namespace h2::meta;
using tlist::Empty;

namespace
{
template <int N>
using IntT = ValueAsType<int, N>;

template <bool B>
using BoolT = ValueAsType<bool, B>;

template <int... Ns>
using IntList = TL<IntT<Ns>...>;

template <typename A, typename B>
struct ValueLess : BoolT<(A::value < B::value)>
{};

} // namespace

// Testing Sort
static_assert(
    EqV<tlist::Sort<Empty, ValueLess>, Empty>(),
    "Sorting the empty list gives the empty list.");

static_assert(
    EqV<tlist::Sort<IntList<13>, ValueLess>, IntList<13>>(),
    "Sorting a singleton gives the singleton.");

static_assert(
    EqV<tlist::Sort<IntList<5, 4, 3, 2, 1>, ValueLess>,
        IntList<1, 2, 3, 4, 5>>(),
    "Sorting a decreasing list; worst case.");

static_assert(
    EqV<tlist::Sort<IntList<1, 2, 3, 4, 5>, ValueLess>,
        IntList<1, 2, 3, 4, 5>>(),
    "Sorting an increasing list; best case.");

static_assert(
    EqV<tlist::Sort<IntList<4, 8, 2, 1, 3, 3, 9, 6>, ValueLess>,
        IntList<1, 2, 3, 3, 4, 6, 8, 9>>(),
    "Sort a random list.");
