////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <h2/utils/Logger.hpp>
#include "./logging/rank_pattern.hpp"
#include "./logging/size_pattern.hpp"
#include "./logging/hostname_pattern.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable

namespace h2
{

void Logger::initialize() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<HostnameFlag>('h');
  formatter->add_flag<SizeFlag>('W');
  formatter->add_flag<RankFlag>('w').set_pattern(
    "[%D %H:%M %z] [%h (Rank %w/%W)] [%^%L%$] %v");

  std::vector<spdlog::sink_ptr> sinks { console_sink };

  auto logger = std::make_shared<spdlog::logger>(
    H2_LOGGER_NAME, sinks.begin(), sinks.end());
  logger->flush_on(spdlog::get_level());
  logger->set_formatter(std::move(formatter));
  spdlog::register_logger(logger);
  load_log_level();
}

void Logger::finalize() {
  spdlog::shutdown();
}

void Logger::load_log_level() {
  // Set the log level to "info" and mylogger to "trace":
  // SPDLOG_LEVEL=info,mylogger=trace && ./example
  spdlog::cfg::load_env_levels();
  //#define H2_LOG_ACTIVE_LEVEL SPDLOG_LEVEL
  // or from command line:
  // ./example SPDLOG_LEVEL=info,mylogger=trace
  //#include "spdlog/cfg/argv.h" // for loading levels from argv
  //spdlog::cfg::load_argv_levels(args, argv);
}

}// namespace h2
