////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "catch2/catch.hpp"
#include <h2/utils/Logger.hpp>
#include "../../src/utils/logger_internals.hpp"
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

    //FIXME: As or with? Should I move to_upper to this function?
    CHECK_THROWS_AS(h2_internal::get_log_level_type("TRSCE"),
                    std::runtime_error);
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("trace"),
                      "Unknown log level: trace");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("debug"),
                      "Unknown log level: debug");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("info"),
                      "Unknown log level: info");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("warn"),
                      "Unknown log level: warn");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("error"),
                      "Unknown log level: error");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("critical"),
                      "Unknown log level: critical");
    CHECK_THROWS_WITH(h2_internal::get_log_level_type("off"),
                      "Unknown log level: off");
  }

  SECTION("Switch from LogLevelType to string")
  {
    //FIXME: How to test throws?
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
    //FIXME: Best way I could think of to test a failing case. This doesn't
    //       really help anything though
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
    CHECK(h2_internal::extract_mask("off") == LogLevelType::OFF);
    CHECK(h2_internal::extract_mask("TRACE|INFO") ==
          (LogLevelType::TRACE | LogLevelType::INFO));

    CHECK(h2_internal::extract_mask("trace|trace|trace|trace|trace|trace|trace")
          == LogLevelType::TRACE);
    CHECK(h2_internal::extract_mask("trace|info|trace|info|trace|info|trace")
          == (LogLevelType::TRACE | LogLevelType::INFO));
    CHECK(h2_internal::extract_mask("debug|crit|debug|crit|debug|crit|debug")
          == (LogLevelType::DEBUG | LogLevelType::CRITICAL));
    CHECK(h2_internal::extract_mask("debug|crit|debug|crit|warn|crit|debug")
          == (LogLevelType::DEBUG | LogLevelType::CRITICAL | LogLevelType::WARN));
    CHECK_THROWS_AS(h2_internal::extract_mask("trace|debug|info|warn|error|crit|trsce|debug|info|warn|error|crit"), std::runtime_error);
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

    CHECK_THROWS_AS(h2_internal::extract_level("debig"), std::runtime_error);
    CHECK_THROWS_AS(h2_internal::extract_level("trace|debug"), std::runtime_error);
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
    auto m = h2_internal::get_keys_and_masks("critical,io=warn|critical,training=trace|info|error");

    CHECK(m.at("") == LogLevelType::CRITICAL);
    CHECK(m.at("io") == (LogLevelType::WARN | LogLevelType::CRITICAL));
    CHECK(m.at("training") == (LogLevelType::TRACE | LogLevelType::INFO |
                               LogLevelType::ERROR));
  }

  SECTION("Get map with keys and levels")
  {
    auto m = h2_internal::get_keys_and_levels("critical,io=warn,training=debug");

    CHECK(m.at("") == LogLevelType::CRITICAL);
    CHECK(m.at("io") == LogLevelType::WARN);
    CHECK(m.at("training") == LogLevelType::DEBUG);
  }

}
