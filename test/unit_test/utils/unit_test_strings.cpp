////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <ostream>

#include "h2/utils/strings.hpp"


using namespace h2;


TEST_CASE("build_string works for empty args", "[utilities][strings]")
{
  std::string s = build_string();
  REQUIRE(s == "");
  REQUIRE(noexcept(build_string()));
}

TEST_CASE("build_string works for empty string", "[utilities][strings]")
{
  std::string s = build_string("");
  REQUIRE(s == "");
  s = build_string("", "", "");
  REQUIRE(s == "");
}

TEST_CASE("build_string works for basic arguments", "[utilities][strings]")
{
  std::string s = build_string("foo");
  REQUIRE(s == "foo");
  s = build_string("foo", " ", "bar");
  REQUIRE(s == "foo bar");
  s = build_string("foo", 42);
  REQUIRE(s == "foo42");
  s = build_string(std::string("foo"));
  REQUIRE(s == "foo");
}

struct TestStruct
{
  int foo;
};

inline std::ostream& operator<<(std::ostream& os, const TestStruct& test)
{
  os << test.foo;
  return os;
}

TEST_CASE("build_string works with user-defined operator<<",
          "[utilities][strings]")
{
  TestStruct test = {42};
  std::string s = build_string("foo", test, " ", "bar ", 43);
  REQUIRE(s == "foo42 bar 43");
}

TEST_CASE("str_toupper works", "[utilities][strings]")
{
  REQUIRE(str_toupper("foo") == "FOO");
  REQUIRE(str_toupper("FOO") == "FOO");
  REQUIRE(str_toupper("fOoBaR") == "FOOBAR");
  REQUIRE(str_toupper("") == "");
  REQUIRE(str_toupper("foo1bar") == "FOO1BAR");
  REQUIRE(str_toupper("123#$%") == "123#$%");
}

TEST_CASE("str_tolower works", "[utilities][strings]")
{
  REQUIRE(str_tolower("FOO") == "foo");
  REQUIRE(str_tolower("foo") == "foo");
  REQUIRE(str_tolower("fOoBaR") == "foobar");
  REQUIRE(str_tolower("") == "");
  REQUIRE(str_tolower("FOO1BAR") == "foo1bar");
  REQUIRE(str_tolower("123#$%") == "123#$%");
}

TEST_CASE("from_string works for strings", "[utilities][strings]")
{
  REQUIRE(from_string("foobar") == "foobar");
  REQUIRE(from_string<std::string>("456foobar123") == "456foobar123");
}

TEST_CASE("from_string works for integers", "[utilities][strings]")
{
  // Note: Some numeric_limits math may not work on really strange
  // systems (e.g., sizeof(char) == sizeof(long long) or something).
  // We probably won't encounter those. (Famous last words.)

  // Note: Conversion to unsigned integers of negative numbers is well-
  // defined using integer wraparound rules. However, this is not well-
  // tested here.

  // char:
  REQUIRE(from_string<char>("42") == 42);
  REQUIRE_THROWS(from_string<char>("foo"));
  REQUIRE_THROWS(from_string<char>(""));
  REQUIRE_THROWS(from_string<char>(
      std::to_string(as<long long>(std::numeric_limits<char>::max()) + 1ll)));
  if constexpr (std::numeric_limits<char>::is_signed)
  {
    REQUIRE(from_string<char>("-42") == -42);
  }

  // signed char:
  REQUIRE(from_string<signed char>("42") == 42);
  REQUIRE(from_string<signed char>("-42") == -42);
  REQUIRE_THROWS(from_string<signed char>("foo"));
  REQUIRE_THROWS(from_string<signed char>(""));
  REQUIRE_THROWS(from_string<signed char>(
                   std::to_string(as<long long>(std::numeric_limits<signed char>::max()) + 1ll)));
  REQUIRE_THROWS(from_string<signed char>(std::to_string(
      as<long long>(std::numeric_limits<signed char>::min()) - 1ll)));

  // unsigned char:
  REQUIRE(from_string<unsigned char>("42") == 42);
  REQUIRE_THROWS(from_string<unsigned char>("foo"));
  REQUIRE_THROWS(from_string<unsigned char>(""));
  REQUIRE_THROWS(from_string<unsigned char>(std::to_string(
      as<long long>(std::numeric_limits<unsigned char>::max()) + 1ll)));

  // short:
  REQUIRE(from_string<short>("42") == 42);
  REQUIRE(from_string<short>("-42") == -42);
  REQUIRE_THROWS(from_string<short>("foo"));
  REQUIRE_THROWS(from_string<short>(""));
  REQUIRE_THROWS(from_string<short>(
      std::to_string(as<long long>(std::numeric_limits<short>::max()) + 1ll)));
  REQUIRE_THROWS(from_string<short>(
      std::to_string(as<long long>(std::numeric_limits<short>::min()) - 1ll)));

  // unsigned short:
  REQUIRE(from_string<unsigned short>("42") == 42);
  REQUIRE_THROWS(from_string<unsigned short>("foo"));
  REQUIRE_THROWS(from_string<unsigned short>(""));
  REQUIRE_THROWS(from_string<unsigned short>(
      std::to_string(as<long long>(std::numeric_limits<unsigned short>::max()) + 1ll)));

  // int:
  REQUIRE(from_string<int>("42") == 42);
  REQUIRE(from_string<int>("-42") == -42);
  REQUIRE_THROWS(from_string<int>("foo"));
  REQUIRE_THROWS(from_string<int>(""));
  REQUIRE_THROWS(from_string<int>(
      std::to_string(as<long long>(std::numeric_limits<int>::max()) + 1ll)));
  REQUIRE_THROWS(from_string<int>(
      std::to_string(as<long long>(std::numeric_limits<int>::min()) - 1ll)));

  // unsigned int:
  REQUIRE(from_string<unsigned int>("42") == 42);
  REQUIRE_THROWS(from_string<unsigned int>("foo"));
  REQUIRE_THROWS(from_string<unsigned int>(""));
  REQUIRE_THROWS(from_string<unsigned int>(
      std::to_string(as<long long>(std::numeric_limits<unsigned int>::max()) + 1ll)));

  // No further testing of max/min due to overflow concerns.

  // long:
  REQUIRE(from_string<long>("42") == 42);
  REQUIRE(from_string<long>("-42") == -42);
  REQUIRE_THROWS(from_string<long>("foo"));
  REQUIRE_THROWS(from_string<long>(""));

  // unsigned long:
  REQUIRE(from_string<unsigned long>("42") == 42);
  REQUIRE_THROWS(from_string<unsigned long>("foo"));
  REQUIRE_THROWS(from_string<unsigned long>(""));

  // long long:
  REQUIRE(from_string<long long>("42") == 42);
  REQUIRE_THROWS(from_string<long long>("foo"));
  REQUIRE_THROWS(from_string<long long>(""));

  // unsigned long long:
  REQUIRE(from_string<unsigned long long>("42") == 42);
  REQUIRE_THROWS(from_string<unsigned long long>("foo"));
  REQUIRE_THROWS(from_string<unsigned long long>(""));
}

TEST_CASE("from_string works for bools", "[utilities][strings]")
{
  REQUIRE(from_string<bool>("1") == true);
  REQUIRE(from_string<bool>("42") == true);
  REQUIRE(from_string<bool>("tRue") == true);
  REQUIRE(from_string<bool>("0") == false);
  REQUIRE(from_string<bool>("faLsE") == false);
  REQUIRE_THROWS(from_string<bool>(""));
  REQUIRE_THROWS(from_string<bool>("foo"));
}

TEST_CASE("from_string works for floating point values", "[utilities][strings]")
{
  // float:
  REQUIRE(from_string<float>("42") == 42.0f);
  REQUIRE(from_string<float>("3.14") == 3.14f);
  REQUIRE(from_string<float>("-inf")
          == -std::numeric_limits<float>::infinity());
  REQUIRE(std::isnan(from_string<float>("nan")));
  REQUIRE_THROWS(from_string<float>(""));
  REQUIRE_THROWS(from_string<float>("foo"));
  REQUIRE_THROWS(from_string<float>(
      std::to_string(std::numeric_limits<long double>::max())));

  // double:
  REQUIRE(from_string<double>("42") == 42.0);
  REQUIRE(from_string<double>("3.14") == 3.14);
  REQUIRE(from_string<double>("-inf")
          == -std::numeric_limits<double>::infinity());
  REQUIRE(std::isnan(from_string<double>("nan")));
  REQUIRE_THROWS(from_string<double>(""));
  REQUIRE_THROWS(from_string<double>("foo"));

  // This is failing on ppc64le. It seems that any overflow to double
  // gets returned as DBL_MAX and errno doesn't actually get set.
  // Until we sort this out, we just skip this for now.
  // if constexpr (std::numeric_limits<long double>::max()
  //               > std::numeric_limits<double>::max()) {
  //   REQUIRE_THROWS(from_string<double>(
  //       std::to_string(std::numeric_limits<long double>::max())));
  // }

  // long double:
  REQUIRE(from_string<long double>("42") == 42.0l);
  REQUIRE(from_string<long double>("3.14") == 3.14l);
  REQUIRE(from_string<long double>("-inf")
          == -std::numeric_limits<long double>::infinity());
  REQUIRE(std::isnan(from_string<long double>("nan")));
  REQUIRE_THROWS(from_string<long double>(""));
  REQUIRE_THROWS(from_string<long double>("foo"));
}
