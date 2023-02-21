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

  /*FIXME: Should these be tested?
    ::spdlog::sink_ptr make_file_sink(std::string const& sinkname);

    ::spdlog::sink_ptr get_file_sink(std::string const& sinkname);

    std::unique_ptr<::spdlog::pattern_formatter> make_h2_formatter(
    std::string const& pattern_prefix);

    std::shared_ptr<::spdlog::logger> make_logger(std::string name,
    std::string const& sink_name,
    std::string const& pattern_prefix);
  */


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

    CHECK_THROWS(h2_internal::get_log_level_type("TYPO"));
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

  SECTION("String parsing functions")
  {
    std::string t = "upperCase";
    CHECK(h2_internal::to_upper(t) == "UPPERCASE");
    t = " leadingSpace";
    CHECK(h2_internal::trim(t) == "leadingSpace");
    t = "trailingSpace ";
    CHECK(h2_internal::trim(t) == "trailingSpace");
  }

  SECTION("Extract mask from string varible")
  {
    CHECK(h2_internal::extract_mask("trace|debug") == (0b00000001|0b00000010));
    CHECK(h2_internal::extract_mask("info|warn") == (0b00000100|0b00001000));
    CHECK(h2_internal::extract_mask("error|critical") == (0b00010000|0b00100000));
    CHECK(h2_internal::extract_mask("off") == (0b00000000));
    CHECK(h2_internal::extract_mask("TRACE|INFO") == (0b00000001|0b00000100));
  }

  SECTION("Extract level from string variable")
  {
    CHECK(h2_internal::extract_level("trace") == LogLevelType::TRACE);
    CHECK(h2_internal::extract_level("debug") == LogLevelType::DEBUG);
    CHECK(h2_internal::extract_level("trace|debug") == LogLevelType::DEBUG);
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
    auto m = h2_internal::get_keys_and_masks("LOG_MASK=critical, io=critical|warn, training=trace|info|error");

    //CHECK(m.at("") == 0b00100000);
    CHECK(m.at("io") == 0b00101000);
    CHECK(m.at("training") == 0b00010101);
  }

  SECTION("Get map with keys and levels")
  {
    auto m = h2_internal::get_keys_and_levels("LOG_MASK=critical, io=warn, training=debug");

    //CHECK(m.at("") == LogLevelType::CRITICAL);
    CHECK(m.at("io") == LogLevelType::WARN);
    CHECK(m.at("training") == LogLevelType::DEBUG);
  }

}
