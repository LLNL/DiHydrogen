////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/Core.hpp"
#include "h2/meta/typelist/Repeat.hpp"

using namespace h2::meta;

static_assert(
    Eq<tlist::Repeat<int, 0>, tlist::Empty>,
    "Repeating zero times gives an empty list.");

static_assert(
    Eq<tlist::Repeat<int, 3>, TL<int, int, int>>,
    "Repeating a nonzero times gives the correct type list.");
