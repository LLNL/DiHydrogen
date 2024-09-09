////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Reverse.hpp"

using namespace h2::meta;

static_assert(Eq<tlist::Reverse<tlist::Empty>, tlist::Empty>,
              "Reversing an empty tlist is valid and returns an empty tlist.");

static_assert(Eq<tlist::Reverse<TL<int>>, TL<int>>,
              "Reversing a single-element list returns a single-element list.");

static_assert(
  Eq<tlist::Reverse<TL<char, short, int, long>>, TL<long, int, short, char>>,
  "Reversing a nontrivial list returns the reversed list.");
