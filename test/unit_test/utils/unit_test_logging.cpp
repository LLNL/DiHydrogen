////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>

#include "h2/utils/Logger.hpp"
#include "../src/utils/logger_internals.hpp"

#include <cstdlib>
#include <iostream>

TEST_CASE("Testing the internal functions used by the logging class",
          "[logging][utilities]")
{
  using LogLevelType = h2::Logger::LogLevelType;

  SECTION("Switch from string to LogLevelType")
  {
    CHECK(h2_internal::get_log_level_type("TRACE") == LogLevelType::TRACE);
    CHECK(h2_internal::get_log_level_type("DEBUG") == LogLevelType::DEBUG);
    CHECK(h2_internal::get_log_level_type("INFO") == LogLevelType::INFO);
    CHECK(h2_internal::get_log_level_type("WARN") == LogLevelType::WARN);
    CHECK(h2_internal::get_log_level_type("WARNING") == LogLevelType::WARN);
    CHECK(h2_internal::get_log_level_type("ERR") == LogLevelType::ERROR);
    CHECK(h2_internal::get_log_level_type("ERROR") == LogLevelType::ERROR);
    CHECK(h2_internal::get_log_level_type("CRITICAL") == LogLevelType::CRITICAL);
    CHECK(h2_internal::get_log_level_type("OFF") == LogLevelType::OFF);

    CHECK_THROWS_WITH(h2_internal::get_log_level_type("TRSCE"),
                      "Invalid log level: TRSCE");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("trace"),
                      "Invalid log level: trace");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("debug"),
                      "Invalid log level: debug");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("info"),
                      "Invalid log level: info");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("warn"),
                      "Invalid log level: warn");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("error"),
                      "Invalid log level: error");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("critical"),
                      "Invalid log level: critical");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("off"),
                      "Invalid log level: off");
  }

  SECTION("Switch from LogLevelType to string")
  {
    CHECK(h2_internal::get_log_level_string(LogLevelType::TRACE) == "TRACE");
    CHECK(h2_internal::get_log_level_string(LogLevelType::DEBUG) == "DEBUG");
    CHECK(h2_internal::get_log_level_string(LogLevelType::INFO) == "INFO");
    CHECK(h2_internal::get_log_level_string(LogLevelType::WARN) == "WARN");
    CHECK(h2_internal::get_log_level_string(LogLevelType::ERROR) == "ERROR");
    CHECK(h2_internal::get_log_level_string(LogLevelType::CRITICAL) == "CRITICAL");
    CHECK(h2_internal::get_log_level_string(LogLevelType::OFF) == "OFF");
  }

  SECTION("To uppercase")
  {
    std::string s = "uppercase";
    CHECK(h2_internal::to_upper(s) == "UPPERCASE");
    s = "uPpErCaSe";
    CHECK(h2_internal::to_upper(s) == "UPPERCASE");
    s = "UPPERCASE";
    CHECK(h2_internal::to_upper(s) == "UPPERCASE");
  }

  SECTION("Trim whitespace")
  {
    std::string s = " leadingSpace";
    CHECK(h2_internal::trim(s) == "leadingSpace");
    s = "trailingSpace ";
    CHECK(h2_internal::trim(s) == "trailingSpace");
    s = " leadAndTrailing ";
    CHECK(h2_internal::trim(s) == "leadAndTrailing");
    s = "        manySpaces         ";
    CHECK(h2_internal::trim(s) == "manySpaces");
    s = "middle spaces";
    CHECK_FALSE(h2_internal::trim(s) == "middlespaces");
  }

  SECTION("Extract mask from string varible")
  {
    CHECK(h2_internal::extract_mask("TRACE") == LogLevelType::TRACE);
    CHECK(h2_internal::extract_mask("trace") == LogLevelType::TRACE);
    CHECK(h2_internal::extract_mask("DEBUG") == LogLevelType::DEBUG);
    CHECK(h2_internal::extract_mask("debug") == LogLevelType::DEBUG);
    CHECK(h2_internal::extract_mask("INFO") == LogLevelType::INFO);
    CHECK(h2_internal::extract_mask("info") == LogLevelType::INFO);
    CHECK(h2_internal::extract_mask("WARN") == LogLevelType::WARN);
    CHECK(h2_internal::extract_mask("warn") == LogLevelType::WARN);
    CHECK(h2_internal::extract_mask("WARNING") == LogLevelType::WARN);
    CHECK(h2_internal::extract_mask("warning") == LogLevelType::WARN);
    CHECK(h2_internal::extract_mask("ERR") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_mask("err") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_mask("ERROR") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_mask("error") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_mask("CRITICAL") == LogLevelType::CRITICAL);
    CHECK(h2_internal::extract_mask("critical") == LogLevelType::CRITICAL);
    CHECK(h2_internal::extract_mask("OFF") == LogLevelType::OFF);
    CHECK(h2_internal::extract_mask("off") == LogLevelType::OFF);

    CHECK(h2_internal::extract_mask("trace|debug") ==
          (LogLevelType::TRACE | LogLevelType::DEBUG));
    CHECK(h2_internal::extract_mask("trace|debug|info") ==
          (LogLevelType::TRACE | LogLevelType::DEBUG | LogLevelType::INFO));
    CHECK(h2_internal::extract_mask("info|warn") ==
          (LogLevelType::INFO | LogLevelType::WARN));
    CHECK(h2_internal::extract_mask("info|warn|error") ==
          (LogLevelType::INFO | LogLevelType::WARN |LogLevelType::ERROR));
    CHECK(h2_internal::extract_mask("warn|error") ==
          (LogLevelType::WARN | LogLevelType::ERROR));
    CHECK(h2_internal::extract_mask("warn|error|critical") ==
          (LogLevelType::WARN | LogLevelType::ERROR | LogLevelType::CRITICAL));
    CHECK(h2_internal::extract_mask("trace|debug|info|warn|error|critical") ==
          (LogLevelType::TRACE | LogLevelType::DEBUG | LogLevelType::INFO |
           LogLevelType::WARN | LogLevelType::ERROR | LogLevelType::CRITICAL));
    CHECK(h2_internal::extract_mask("critical|off") ==
          (LogLevelType::CRITICAL | LogLevelType::OFF));

    CHECK(h2_internal::extract_mask("trace|trace|trace|trace|trace|trace|trace")
          == LogLevelType::TRACE);
    CHECK(h2_internal::extract_mask("trace|info|trace|info|trace|info|trace")
          == (LogLevelType::TRACE | LogLevelType::INFO));
    CHECK(h2_internal::extract_mask("debug|crit|debug|crit|debug|crit|debug")
          == (LogLevelType::DEBUG | LogLevelType::CRITICAL));
    CHECK(h2_internal::extract_mask("debug|crit|debug|crit|warn|crit|debug")
          == (LogLevelType::DEBUG | LogLevelType::CRITICAL | LogLevelType::WARN));

    CHECK_THROWS_WITH(h2_internal::extract_mask("trace|debug|info|warn|error|crit|trsce|debug|info|warn|error|crit"), "Invalid log level: TRSCE");
  }

  SECTION("Extract level from string variable")
  {
    CHECK(h2_internal::extract_level("TRACE") == LogLevelType::TRACE);
    CHECK(h2_internal::extract_level("trace") == LogLevelType::TRACE);
    CHECK(h2_internal::extract_level("DEBUG") == LogLevelType::DEBUG);
    CHECK(h2_internal::extract_level("debug") == LogLevelType::DEBUG);
    CHECK(h2_internal::extract_level("INFO") == LogLevelType::INFO);
    CHECK(h2_internal::extract_level("info") == LogLevelType::INFO);
    CHECK(h2_internal::extract_level("WARN") == LogLevelType::WARN);
    CHECK(h2_internal::extract_level("warn") == LogLevelType::WARN);
    CHECK(h2_internal::extract_level("WARNING") == LogLevelType::WARN);
    CHECK(h2_internal::extract_level("warning") == LogLevelType::WARN);
    CHECK(h2_internal::extract_level("ERR") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_level("err") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_level("ERROR") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_level("error") == LogLevelType::ERROR);
    CHECK(h2_internal::extract_level("CRITICAL") == LogLevelType::CRITICAL);
    CHECK(h2_internal::extract_level("critical") == LogLevelType::CRITICAL);
    CHECK(h2_internal::extract_level("OFF") == LogLevelType::OFF);
    CHECK(h2_internal::extract_level("off") == LogLevelType::OFF);

    CHECK_THROWS_WITH(h2_internal::extract_level("debig"),
                      "Invalid log level: DEBIG");
    CHECK_THROWS_WITH(h2_internal::extract_level("trace|debug"),
                      "Invalid log level: TRACE|DEBUG");
  }

  SECTION("Extract key and value pair from string")
  {
    auto p = h2_internal::extract_key_and_val('=', "trace|debug");
    CHECK(p.first == "");
    CHECK(p.second == "trace|debug");

    p = h2_internal::extract_key_and_val('=', "io=trace|debug");
    CHECK(p.first == "io");
    CHECK(p.second == "trace|debug");
  }

  SECTION("Get map with keys and masks")
  {
    auto m = h2_internal::get_keys_and_masks(
      "critical,io=warn|critical,training=trace|info|error");
    CHECK(m.at("") == LogLevelType::CRITICAL);
    CHECK(m.at("io") == (LogLevelType::WARN | LogLevelType::CRITICAL));
    CHECK(m.at("training") == (LogLevelType::TRACE | LogLevelType::INFO |
                               LogLevelType::ERROR));

    m = h2_internal::get_keys_and_masks(
      ",io=error|critical,training=info|warn|error");
    CHECK(m.count("") == 0);
    CHECK(m.at("io") == (LogLevelType::ERROR | LogLevelType::CRITICAL));
    CHECK(m.at("training") == (LogLevelType::INFO | LogLevelType::WARN |
                               LogLevelType::ERROR));

    m = h2_internal::get_keys_and_masks(
      ",,,,io=trace,,,training=warn|info,,,,");
    CHECK(m.count("") == 0);
    CHECK(m.at("io") == (LogLevelType::TRACE));
    CHECK(m.at("training") == (LogLevelType::WARN | LogLevelType::INFO));

    m = h2_internal::get_keys_and_masks(
      "critical|error,warn,io=warn|critical,training=trace|info|error");
    CHECK(m.at("") == LogLevelType::WARN);
    CHECK(m.at("io") == (LogLevelType::WARN | LogLevelType::CRITICAL));
    CHECK(m.at("training") == (LogLevelType::TRACE | LogLevelType::INFO |
                               LogLevelType::ERROR));

    m = h2_internal::get_keys_and_masks(
      "critical|error,io=warn|error,training=debug|warn|critical,trace");
    CHECK(m.at("") == LogLevelType::TRACE);
    CHECK(m.at("io") == (LogLevelType::WARN | LogLevelType::ERROR));
    CHECK(m.at("training") == (LogLevelType::DEBUG | LogLevelType::WARN |
                               LogLevelType::CRITICAL));

    m = h2_internal::get_keys_and_masks(
      "critical|error,io=warn|error,training=debug|warn|critical,trace,info,error");
    CHECK(m.at("") == LogLevelType::ERROR);
    CHECK(m.at("io") == (LogLevelType::WARN | LogLevelType::ERROR));
    CHECK(m.at("training") == (LogLevelType::DEBUG | LogLevelType::WARN |
                               LogLevelType::CRITICAL));
  }

  SECTION("Get map with keys and levels")
  {
    auto m = h2_internal::get_keys_and_levels("critical,io=warn,training=debug");

    CHECK(m.at("") == LogLevelType::CRITICAL);
    CHECK(m.at("io") == LogLevelType::WARN);
    CHECK(m.at("training") == LogLevelType::DEBUG);

    m = h2_internal::get_keys_and_levels(",io=error,training=info");
    CHECK(m.count("") == 0);
    CHECK(m.at("io") == LogLevelType::ERROR);
    CHECK(m.at("training") == LogLevelType::INFO);

    m = h2_internal::get_keys_and_levels(
      ",,,,io=trace,,,training=warn,,,,");
    CHECK(m.count("") == 0);
    CHECK(m.at("io") == LogLevelType::TRACE);
    CHECK(m.at("training") == LogLevelType::WARN);

    m = h2_internal::get_keys_and_levels("error,warn,io=critical,training=trace");
    CHECK(m.at("") == LogLevelType::WARN);
    CHECK(m.at("io") == LogLevelType::CRITICAL);
    CHECK(m.at("training") == LogLevelType::TRACE);

    m = h2_internal::get_keys_and_levels("critical,io=debug,training=trace,trace");
    CHECK(m.at("") == LogLevelType::TRACE);
    CHECK(m.at("io") == LogLevelType::DEBUG);
    CHECK(m.at("training") == LogLevelType::TRACE);

    m = h2_internal::get_keys_and_levels(
      "error,io=warn,training=debug,trace,info,error");
    CHECK(m.at("") == LogLevelType::ERROR);
    CHECK(m.at("io") == LogLevelType::WARN);
    CHECK(m.at("training") == LogLevelType::DEBUG);
  }
}
