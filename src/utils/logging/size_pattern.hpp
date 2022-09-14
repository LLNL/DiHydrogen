////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_UTILS_LOGGING_SIZE_PATTERN_HPP_INCLUDED
#define H2_UTILS_LOGGING_SIZE_PATTERN_HPP_INCLUDED

#include "spdlog/pattern_formatter.h"

namespace
{

class SizeFlag : public spdlog::custom_flag_formatter
{
public:
  void format(const spdlog::details::log_msg &, const std::tm &,
              spdlog::memory_buf_t &dest) override
  {
    std::string size = std::to_string(get_local_size());
    dest.append(size.data(), size.data() + size.size());
  }

  std::unique_ptr<custom_flag_formatter> clone() const override
  {
    return spdlog::details::make_unique<SizeFlag>();
  }

  /** Attempt to identify world size from the environment. */
  int get_local_size()
  {
    char* env = std::getenv("MV2_COMM_WORLD_SIZE");
    if (!env)
      env = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!env) {
      // Cannot determine world size
      env = "-1";
    }
    return std::atoi(env);
  }

};// class SizeFlag

} // namespace h2

#endif // H2_UTILS_LOGGING_SIZE_PATTERN_HPP_INCLUDED
