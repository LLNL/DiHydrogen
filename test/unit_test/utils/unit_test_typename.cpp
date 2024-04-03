////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include "h2/utils/typename.hpp"


using namespace h2;


TEST_CASE("TypeName works for built-in types", "[utilities][typename]")
{
  REQUIRE(TypeName<bool>() == "bool");
  REQUIRE(TypeName<char>() == "char");
  REQUIRE(TypeName<unsigned char>() == "unsigned char");
  REQUIRE(TypeName<int>() == "int");
  REQUIRE(TypeName<float>() == "float");
  REQUIRE(TypeName<double>() == "double");
  REQUIRE(TypeName<long double>() == "long double");
}

struct TestStruct {};
template <typename T> struct TestTemplateStruct {};

TEST_CASE("TypeName works for complicated types", "[utilities][typename]")
{
  REQUIRE(TypeName<TestStruct>() == "TestStruct");
  REQUIRE(TypeName<TestTemplateStruct<int>>() == "TestTemplateStruct<int>");
  REQUIRE(TypeName <TestTemplateStruct<TestStruct>>()
          == "TestTemplateStruct<TestStruct>");
}
