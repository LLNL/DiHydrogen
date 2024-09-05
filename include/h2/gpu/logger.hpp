////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2_config.hpp"

#include <spdlog/spdlog.h>

// We can ignore the SPDLOG level and manage it here.

#define H2_LOG_LEVEL_TRACE SPDLOG_LEVEL_TRACE
#define H2_LOG_LEVEL_DEBUG SPDLOG_LEVEL_DEBUG
#define H2_LOG_LEVEL_INFO SPDLOG_LEVEL_INFO
#define H2_LOG_LEVEL_WARN SPDLOG_LEVEL_WARN
#define H2_LOG_LEVEL_ERROR SPDLOG_LEVEL_ERROR
#define H2_LOG_LEVEL_CRITICAL SPDLOG_LEVEL_CRITICAL
#define H2_LOG_LEVEL_OFF SPDLOG_LEVEL_OFF

#ifndef H2_GPU_LOG_ACTIVE_LEVEL
#define H2_GPU_LOG_ACTIVE_LEVEL H2_LOG_LEVEL_TRACE
#endif

#define H2_GPU_LOG(level, ...)                                                 \
    ::h2::gpu::logger().log(                                                   \
        ::spdlog::source_loc{__FILE__, __LINE__, H2_PRETTY_FUNCTION},          \
        level,                                                                 \
        __VA_ARGS__)

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_TRACE
#define H2_GPU_TRACE(...) H2_GPU_LOG(::spdlog::level::trace, __VA_ARGS__)
#else
#define H2_GPU_TRACE(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_TRACE

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_DEBUG
#define H2_GPU_DEBUG(...) H2_GPU_LOG(::spdlog::level::debug, __VA_ARGS__)
#else
#define H2_GPU_DEBUG(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_DEBUG

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_INFO
#define H2_GPU_INFO(...) H2_GPU_LOG(::spdlog::level::info, __VA_ARGS__)
#else
#define H2_GPU_INFO(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_INFO

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_WARN
#define H2_GPU_WARN(...) H2_GPU_LOG(::spdlog::level::warn, __VA_ARGS__)
#else
#define H2_GPU_WARN(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_WARN

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_ERROR
#define H2_GPU_ERROR(...) H2_GPU_LOG(::spdlog::level::err, __VA_ARGS__)
#else
#define H2_GPU_ERROR(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_ERROR

#if H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_CRITICAL
#define H2_GPU_CRITICAL(...) H2_GPU_LOG(::spdlog::level::critical, __VA_ARGS__)
#else
#define H2_GPU_CRITICAL(...) (void) 0
#endif // H2_GPU_LOG_ACTIVE_LEVEL <= H2_LOG_LEVEL_CRITICAL

namespace h2
{
namespace gpu
{

/** @brief Get the spdlog::logger being used to track the GPU logs. */
spdlog::logger& logger();

} // namespace gpu
} // namespace h2
