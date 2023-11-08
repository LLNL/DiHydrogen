////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "spdlog/spdlog.h"

#pragma once
#ifndef H2_UTILS_LOGGER_INTERNALS_HPP_INCLUDED
#define H2_UTILS_LOGGER_INTERNALS_HPP_INCLUDED

namespace h2_internal
{
using LevelMapType = std::unordered_map<std::string, h2::Logger::LogLevelType>;
using MaskMapType = std::unordered_map<std::string, unsigned char>;

::spdlog::sink_ptr make_file_sink(std::string const& sinkname);

::spdlog::sink_ptr get_file_sink(std::string const& sinkname);

std::unique_ptr<::spdlog::pattern_formatter> make_h2_formatter(
  std::string const& pattern_prefix);

std::shared_ptr<::spdlog::logger> make_logger(std::string name,
                                              std::string const& sink_name,
                                              std::string const& pattern_prefix);

h2::Logger::LogLevelType get_log_level_type(std::string const& level);

std::string get_log_level_string(h2::Logger::LogLevelType const& level);

std::string& to_upper(std::string &str);

std::string& trim(std::string &str);

unsigned char extract_mask(std::string levels);

h2::Logger::LogLevelType extract_level(std::string level);

std::pair<std::string, std::string> extract_key_and_val(
  char delim, const std::string &str);

MaskMapType get_keys_and_masks(std::string const& str);

LevelMapType get_keys_and_levels(std::string const& str);

}

#endif // H2_UTILS_LOGGER_INTERNALS_HPP_INCLUDED
