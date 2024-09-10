////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/meta/typelist/print.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

TEST_CASE("Printing empty typelists works", "[typelist]")
{
  REQUIRE(meta::tlist::print(meta::tlist::Empty{}) == "");
}

TEST_CASE("Printing single-entry typelists works", "[typelist]")
{
  REQUIRE(meta::tlist::print(meta::TL<int>{}) == "int");
}

TEST_CASE("Printing multi-entry typelists works", "[typelist]")
{
  REQUIRE(meta::tlist::print(meta::TL<int, char, float>{})
          == "int, char, float");
}
