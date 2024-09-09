////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/environment_vars.hpp"

#include <stdlib.h>

#include <tuple>

#include <catch2/catch_test_macros.hpp>

using namespace h2;

// Set an environment variable on construction, and unset it afterward.
struct RAIIEnvVar
{
  RAIIEnvVar(std::string name, std::string value) : env_name(name)
  {
    if (getenv(name.c_str()))
    {
      throw std::runtime_error(
        std::string("Attempt to set environment variable ") + name
        + std::string(" but it is already set in the environment"));
    }

    if (setenv(name.c_str(), value.c_str(), 0) != 0)
    {
      throw std::runtime_error(
        std::string("Failed to set environemtn variable ") + name);
    }
  }

  ~RAIIEnvVar() { unsetenv(env_name.c_str()); }

  std::string env_name;
};

TEST_CASE("Raw env vars work", "[utilities][environment_vars]")
{
  RAIIEnvVar env_manager("TEST_FOO_BAR", "1");

  REQUIRE(env::raw::exists("TEST_FOO_BAR"));
  REQUIRE_FALSE(env::raw::exists("TEST_BAR_FOO"));
  REQUIRE(env::raw::get_raw("TEST_FOO_BAR") == "1");
  REQUIRE(env::raw::get_raw("TEST_BAR_FOO") == "");
  REQUIRE(env::raw::get<int>("TEST_FOO_BAR") == 1);
}

TEST_CASE("H2 env vars work", "[utilities][environment_vars]")
{
  // Note: Due to caching, we have only one test case for this.

  // Ensure the variable we're testing isn't already set.
  std::string const h2_env1 = "TEST_VAR1";
  std::string const h2_env2 = "TEST_VAR2";
  if (env::raw::exists("H2_" + h2_env1) || env::raw::exists("H2_" + h2_env2))
  {
    throw std::runtime_error("Test variable already set in env");
  }

  // Check var1 for its default value. (Immediately cached.)
  REQUIRE_FALSE(env::exists(h2_env1));
  REQUIRE(env::get_raw(h2_env1) == "0");
  REQUIRE(env::get<int>(h2_env1) == 0);

  // Check var2, making sure it is first set.
  RAIIEnvVar env_manager("H2_" + h2_env2, "1");
  REQUIRE(env::exists(h2_env2));
  REQUIRE(env::get_raw(h2_env2) == "1");
  REQUIRE(env::get<int>(h2_env2) == 1);
}
