////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Flatten.hpp"

using namespace h2::meta;

struct A;
struct B;
struct C;

using ResultTList = TL<A, B, C, A, A, B, A>;

// Empty
static_assert(EqV<tlist::Flatten<tlist::Nil>, tlist::Nil>(),
              "Flatten nil typelist returns nil.");

// Already flat
static_assert(EqV<tlist::Flatten<A, B, C, A, A, B, A>, ResultTList>(),
              "Flattening a flat list is a NOOP");
static_assert(EqV<tlist::Flatten<ResultTList>, ResultTList>(),
              "Flattening a flat TL is a NOOP");

// No nesting
static_assert(
    EqV<tlist::Flatten<TL<A, B, C>, A, TL<A, B>, TL<A>>, ResultTList>(),
    "No nested TLs");
static_assert(
    EqV<tlist::Flatten<TL<A>, TL<B>, TL<C>, TL<A>, TL<A>, TL<B>, TL<A>>,
        ResultTList>(),
    "All single-element TLs");

// Nesting TLs
static_assert(
    EqV<tlist::Flatten<TL<TL<A, B, C>, A, TL<A, B>, TL<A>>>, ResultTList>(),
    "All types under one nested TL");
static_assert(
    EqV<tlist::Flatten<TL<TL<A>, TL<B>, TL<C>, TL<A>, TL<A>, TL<B>, TL<A>>>,
        ResultTList>(),
    "All single-element TLs nested under one TL");

static_assert(
    EqV<tlist::Flatten<TL<TL<A, B>, TL<C, TL<A>>>, TL<A, TL<B>, TL<A>>>,
        ResultTList>(),
    "Complex nesting");
