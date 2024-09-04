////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/core/Eq.hpp"
#include "h2/meta/core/IfThenElse.hpp"

using namespace h2::meta;

static_assert(
    EqV<IfThenElse<true, int, float>, int>(), "True selects the first type.");
static_assert(
    EqV<IfThenElse<false, int, float>, float>(),
    "False selects the second type.");
