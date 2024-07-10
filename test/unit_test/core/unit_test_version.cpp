////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "h2_config.hpp"
#include "h2/Version.hpp"

using namespace h2;

// gotta get that coverage
TEST_CASE("Version", "[version][core]")
{
    REQUIRE(Version() == H2_VERSION);
}
