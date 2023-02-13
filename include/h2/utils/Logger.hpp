////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once
#ifndef H2_UTILS_LOGGER_HPP_INCLUDED
#define H2_UTILS_LOGGER_HPP_INCLUDED

#include "spdlog/pattern_formatter.h"
#include "spdlog/spdlog.h"

namespace h2
{
class Logger
{
public:

  enum LogLevelType : unsigned char {
        TRACE = 0x1,
        DEBUG = 0x2,
        INFO = 0x4,
        WARN = 0x8,
        ERROR = 0x10,
        CRITICAL = 0x20,
        OFF = 0x40,
    };// enum class LogLevelType

    /** @brief Logger constructor. Logs to stdout.
     *  @param std::string name Name of logger. (Default = "logger")
     **/
     Logger(std::string name)
       : Logger(std::move(name), "stdout")
    {}
    /** @brief Logger constructor.
     *  @param std::string name Name of logger.
     *  @param std::string sink Name of output/file sink.
     *  @param std::string pattern Pattern for log message tags.
     *  Default = [<Date> <Time> <Timezone>] [<Hostname> <Rank>] [<Log Level>]
     **/
    Logger(std::string name, std::string sink,
         std::string pattern = "[%D %H:%M %z] [%h (Rank %w/%W)] [%^%L%$] %v");
    Logger() = default;
    ~Logger() {}

    /** @brief Return name of logger. */
    std::string name() const { return m_logger->name(); }

    /** @brief Get spdlod::logger. */
    ::spdlog::logger& get() { return *m_logger; }

    /** @brief Set log level (Hierarchical logging levels).
     *  @param LogLevelType level Logging level.
     **/
    void set_log_level(LogLevelType level);

    /** @brief Set logging mask.
     *  @param unsigned char mask Logging mask.
     **/
    void set_mask(unsigned char mask);

    /** @brief Check if log message is within set log levels.
     *  @param LogLevelType level Logging level.
     **/
    bool should_log(LogLevelType level) const noexcept;

private:

    std::shared_ptr<::spdlog::logger> m_logger;
    unsigned char m_mask;

}; // class Logger

/** @brief Set log levels for multiple loggers (Hierarchical logging levels).
 *  @param std::vector<Logger*> loggers Vector of Logger pointers.
 *  @param char const* const level_env_var Name of environmental variable.
 **/
void setup_levels(std::vector<Logger*>& loggers,
                  char const* const level_env_var);

/** @brief Set log masks for multiple loggers (Hierarchical logging levels).
 *  @param std::vector<Logger*> loggers Vector of Logger pointers.
 *  @param char const* const level_env_var Name of environmental variable.
 **/
void setup_masks(std::vector<Logger*>& loggers,
                 char const* const mask_env_var);

/** @brief Get spdlog::level type
 *  @param LogLevelType level H2::Logging level.
 **/
spdlog::level::level_enum to_spdlog_level(Logger::LogLevelType level);

} // namespace h2

#endif // H2_UTILS_LOGGER_HPP_INCLUDED
