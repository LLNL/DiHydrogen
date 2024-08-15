////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/core/types.hpp"

#include "../tensor/utils.hpp"

using namespace h2;


TEMPLATE_LIST_TEST_CASE("Runtime type info for compute types works",
                        "[types]",
                        ComputeTypes)
{
  using Type = TestType;

  TypeInfo tinfo = get_h2_type<Type>();
  TypeInfo tinfo_same = get_h2_type<Type>();
  // Pointers are never compute types.
  TypeInfo tinfo_diff = get_h2_type<Type*>();

  REQUIRE(tinfo.get_token() >= 0);
  // All compute types should have nice tokens.
  REQUIRE(tinfo.get_token() < TypeInfo::max_token);
  REQUIRE(tinfo.get_size() == sizeof(Type));
  REQUIRE(*tinfo.get_type_info() == typeid(Type));

  REQUIRE(tinfo == tinfo);
  REQUIRE_FALSE(tinfo != tinfo);
  REQUIRE(tinfo == tinfo_same);
  REQUIRE_FALSE(tinfo != tinfo_same);
  REQUIRE_FALSE(tinfo == tinfo_diff);
  REQUIRE(tinfo != tinfo_diff);
}

struct TestStruct { int foo; };

TEST_CASE("Runtime type info works for any type", "[types]")
{
  SECTION("H2TypeInfo works for non-compute standard types")
  {
    TypeInfo tinfo = get_h2_type<char>();
    // In case something changes and we need to update this.
    static_assert(!IsH2ComputeType_v<char>);
    REQUIRE(tinfo.get_token() >= 0);
    REQUIRE(tinfo.get_size() == sizeof(char));
    REQUIRE(*tinfo.get_type_info() == typeid(char));
  }

  SECTION("H2TypeInfo works for non-compute non-standard types")
  {
    TypeInfo tinfo = get_h2_type<TestStruct>();
    REQUIRE(tinfo.get_token() == TypeInfo::max_token);
    REQUIRE(tinfo.get_size() == sizeof(TestStruct));
    REQUIRE(*tinfo.get_type_info() == typeid(TestStruct));
  }
}
