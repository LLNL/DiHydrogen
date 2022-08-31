////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <h2/utils/Logger.hpp>
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/cfg/env.h"  // support for loading levels from the environment variable

namespace h2
{

void Logger::initialize() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  // FIXME: Custom message format or just default?
  // [Hour:Minute:Second timezone] [colors Log level end-colors] [thread id] msg
  // https://spdlog.docsforge.com/v1.x/3.custom-formatting/#efficiency
  console_sink->set_pattern("[%H:%M:%S %z] [%^%L%$] [thread %t] %v");

  std::vector<spdlog::sink_ptr> sinks { console_sink };

  auto logger = std::make_shared<spdlog::logger>(
    H2_LOGGER_NAME, sinks.begin(), sinks.end());
  load_log_level();
  logger->flush_on(spdlog::get_level());
  spdlog::register_logger(logger);
}

void Logger::shutdown() {
  spdlog::shutdown();
}

void Logger::load_log_level() {
  //spdlog::set_level(spdlog::level::trace);
  // Set the log level to "info" and mylogger to "trace":
  // SPDLOG_LEVEL=info,mylogger=trace && ./example
  spdlog::cfg::load_env_levels();
  // or from command line:
  // ./example SPDLOG_LEVEL=info,mylogger=trace
  // #include "spdlog/cfg/argv.h" // for loading levels from argv
  // spdlog::cfg::load_argv_levels(args, argv);
}

}// namespace h2
