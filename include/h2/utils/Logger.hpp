////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef H2_UTILS_LOGGER_HPP_INCLUDED
#define H2_UTILS_LOGGER_HPP_INCLUDED

#include "spdlog/spdlog.h"

#define H2_LOGGER_NAME "h2_logger"
// FIXME(KLG): Do we want this?
//#if CONFIGURATION != RELEASE
#define H2_TRACE(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->trace(__VA_ARGS__);}

#define H2_DEBUG(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->debug(__VA_ARGS__);}

#define H2_INFO(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->info(__VA_ARGS__);}

#define H2_ERROR(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->error(__VA_ARGS__);}

#define H2_WARNING(...) if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::warn(__VA_ARGS__);}

#define H2_CRITICAL(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->critical(__VA_ARGS__);}
//#else
//#define H2_TRACE(...)  void(0)
//#endif

namespace h2
{

class Logger
{
public:

  Logger() { initialize(); }
  ~Logger() { shutdown(); }

private:
  void initialize();
  void shutdown();
  void load_log_level();

};// class Logger
} // namespace h2
#endif // H2_UTILS_LOGGER_HPP_INCLUDED
