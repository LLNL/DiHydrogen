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
#include "spdlog/pattern_formatter.h"

#define H2_LOGGER_NAME "h2_logger"

#define H2_LOG_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define H2_LOG_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define H2_LOG_LEVEL_INFO SPDLOG_LEVEL_INFO
#define H2_LOG_LEVEL_WARN SPDLOG_LEVEL_WARN
#define H2_LOG_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define H2_LOG_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define H2_LOG_LEVEL_OFF SPDLOG_LEVEL_OFF

#ifndef H2_LOG_ACTIVE_LEVEL
#define H2_LOG_ACTIVE_LEVEL H2_LOG_LEVEL_INFO
#endif

/*
FIXME(KLG): Do I need this?
#define H2_LOG(level, ...)                                                 \
    ::h2::logger().log(                                                   \
        ::spdlog::source_loc{__FILE__, __LINE__, H2_PRETTY_FUNCTION},          \
        level,                                                                 \
        __VA_ARGS__)
*/

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_TRACE
#define H2_TRACE(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->trace(__VA_ARGS__);}
#else
#define H2_TRACE(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_TRACE

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_DEBUG
#define H2_DEBUG(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->debug(__VA_ARGS__);}
#else
#define H2_DEBUG(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_DEBUG

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_INFO
#define H2_INFO(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->info(__VA_ARGS__);}
#else
#define H2_INFO(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_INFO

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_ERROR
#define H2_ERROR(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->error(__VA_ARGS__);}
#else
#define H2_ERROR(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_ERROR

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_WARN
#define H2_WARN(...) if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->warn(__VA_ARGS__);}
#else
#define H2_WARN(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_WARN

#if H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_CRITICAL
#define H2_CRITICAL(...)  if (spdlog::get(H2_LOGGER_NAME) != nullptr) {spdlog::get(H2_LOGGER_NAME)->critical(__VA_ARGS__);}
#else
#define H2_CRITICAL(...) (void) 0
#endif // H2_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_CRITICAL

namespace h2
{

class Logger
{
public:

  Logger() { initialize(); }
  ~Logger() { finalize(); }

private:
  void initialize();
  void finalize();
  void load_log_level();

};// class Logger

class my_formatter_flag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg &, const std::tm &, spdlog::memory_buf_t &dest) override
    {
        std::string some_txt = "custom-flag";
        dest.append(some_txt.data(), some_txt.data() + some_txt.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<my_formatter_flag>();
    }
};

} // namespace h2

#endif // H2_UTILS_LOGGER_HPP_INCLUDED
