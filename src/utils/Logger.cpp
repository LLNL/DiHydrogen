////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/Logger.hpp"

#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include "spdlog/sinks/basic_file_sink.h"

#include <cstdlib>
#include <unordered_map>
#include <sstream>
#include <iostream>

#if __has_include(<mpi.h>)
#define H2_LOGGER_HAS_MPI
#include <mpi.h>
#endif

#if __has_include(<unistd.h>)
#define H2_LOGGER_HAS_UNISTD_H
#include <unistd.h>
#endif

namespace
{
char const* get_first_env(std::initializer_list<char const*> names)
{
    for (auto const& name : names)
    {
        if (char const* ptr = std::getenv(name))
            return ptr;
    }
    return nullptr;
}

class MPIRankFlag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg&,
                const std::tm&,
                spdlog::memory_buf_t& dest) override
    {
        static std::string rank = get_rank_str();
        dest.append(rank.data(), rank.data() + rank.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<MPIRankFlag>();
    }

    static std::string get_rank_str()
    {
        int rank = get_rank_mpi();
        if (rank < 0)
            rank = get_rank_env();
        return (rank >= 0 ? std::to_string(rank) : std::string("?"));
    }

    static int get_rank_mpi()
    {
#ifdef H2_LOGGER_HAS_MPI
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
        {
            int rank = 0;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            return rank;
        }
#endif // H2_LOGGER_HAS_MPI
        return -1;
    }

    static int get_rank_env()
    {
        char const* env = get_first_env({"FLUX_TASK_RANK",
                                         "SLURM_PROCID",
                                         "PMI_RANK",
                                         "MPIRUN_RANK",
                                         "OMPI_COMM_WORLD_RANK",
                                         "MV2_COMM_WORLD_RANK"});
        return env ? std::atoi(env) : -1;
    }
}; // class MPIRankFlag

class MPISizeFlag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg&,
                const std::tm&,
                spdlog::memory_buf_t& dest) override
    {
        static std::string size = get_size_str();
        dest.append(size.data(), size.data() + size.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<MPISizeFlag>();
    }

    static std::string get_size_str()
    {
        int size = get_size_mpi();
        if (size < 0)
            size = get_size_env();
        return (size >= 0 ? std::to_string(size) : std::string("?"));
    }

    static int get_size_mpi()
    {
#ifdef H2_LOGGER_HAS_MPI
        int is_init = 0;
        MPI_Initialized(&is_init);
        if (is_init)
        {
            int size = 0;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
        }
#endif // H2_LOGGER_HAS_MPI
        return -1;
    }

    static int get_size_env()
    {
        char const* env = get_first_env({"FLUX_JOB_SIZE",
                                         "SLURM_NTASKS",
                                         "PMI_SIZE",
                                         "MPIRUN_NTASKS",
                                         "OMPI_COMM_WORLD_SIZE",
                                         "MV2_COMM_WORLD_SIZE"});
        return env ? std::atoi(env) : -1;
    }
}; // class MPISizeFlag

class HostnameFlag final : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg&,
                const std::tm&,
                spdlog::memory_buf_t& dest) override
    {
        static auto const hostname = get_hostname();
        dest.append(hostname.data(), hostname.data() + hostname.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<HostnameFlag>();
    }

#ifdef H2_LOGGER_HAS_UNISTD_H
    static std::string get_hostname()
    {
        char buf[1024];
        if (gethostname(buf, 1024) != 0)
            throw std::runtime_error("gethostname failed.");
        auto end = std::find(buf, buf + 1024, '\0');
        return std::string{buf, end};
    }
#else
    static std::string const& get_hostname()
    {
        static std::string const hostname = "<unknown>";
        return hostname;
    }
#endif // H2_LOGGER_HAS_UNISTD_H

}; // class HostnameFlag

} // namespace

namespace
{
static std::unordered_map<std::string, h2::Logger::LogLevelType>
const log_levels = {
  { "TRACE", h2::Logger::LogLevelType::TRACE },
  { "DEBUG", h2::Logger::LogLevelType::DEBUG },
  { "INFO", h2::Logger::LogLevelType::INFO },
  { "WARN", h2::Logger::LogLevelType::WARN },
  { "WARNING", h2::Logger::LogLevelType::WARN },
  { "ERROR", h2::Logger::LogLevelType::ERROR },
  { "ERR", h2::Logger::LogLevelType::ERROR },
  { "CRITICAL", h2::Logger::LogLevelType::CRITICAL },
  { "OFF", h2::Logger::LogLevelType::OFF } };

static std::unordered_map<h2::Logger::LogLevelType, std::string>
const string_log_levels = {
  { h2::Logger::LogLevelType::TRACE, "TRACE" },
  { h2::Logger::LogLevelType::DEBUG, "DEBUG" },
  { h2::Logger::LogLevelType::INFO, "INFO" },
  { h2::Logger::LogLevelType::WARN, "WARN" },
  { h2::Logger::LogLevelType::WARN, "WARNING" },
  { h2::Logger::LogLevelType::ERROR, "ERROR" },
  { h2::Logger::LogLevelType::ERROR, "ERR" },
  { h2::Logger::LogLevelType::CRITICAL, "CRITICAL" },
  { h2::Logger::LogLevelType::OFF, "OFF" } };

std::unordered_map<std::string, ::spdlog::sink_ptr> sink_map_;

::spdlog::sink_ptr make_file_sink(std::string const& sinkname)
{
    if (sinkname == "stdout")
        return std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    if (sinkname == "stderr")
        return std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    return std::make_shared<spdlog::sinks::basic_file_sink_mt>(sinkname);
}

::spdlog::sink_ptr get_file_sink(std::string const& sinkname)
{
    auto& sink = sink_map_[sinkname];
    if (!sink)
        sink = make_file_sink(sinkname);
    return sink;
}

std::unique_ptr<::spdlog::pattern_formatter> make_h2_formatter(
  std::string pattern)
{
    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<HostnameFlag>('h');
    formatter->add_flag<MPISizeFlag>('W');
    formatter->add_flag<MPIRankFlag>('w');
    formatter->set_pattern(pattern + " %v");

    return formatter;
}

std::shared_ptr<::spdlog::logger> make_logger(std::string name,
                                              std::string const& sink_name,
                                              std::string pattern)
{
    auto logger = std::make_shared<::spdlog::logger>(std::move(name),
                                                     get_file_sink(sink_name));
    logger->set_formatter(make_h2_formatter(pattern));
    ::spdlog::register_logger(logger);
    logger->set_level(::spdlog::level::trace);
    return logger;
}

h2::Logger::LogLevelType get_log_level_type(std::string level)
{
  auto it = log_levels.find(level);
  if (it != log_levels.end()) {
    return it->second;
  }
  else {
    throw std::runtime_error("Invalid log level: " + level); }

}

std::string get_string(h2::Logger::LogLevelType level)
{
  auto it = string_log_levels.find(level);
  if (it != string_log_levels.end()) {
    return it->second;
  }
  else {
    throw std::runtime_error("Invalid log level"); }

}

// convert to uppercase
std::string& to_upper(std::string &str)
{
  std::transform(
    str.begin(), str.end(), str.begin(), [](char ch) {
      return static_cast<char>((ch >= 'a' && ch <= 'z') ?
                               ch - ('a' - 'A') : ch); });
  return str;
}

// trim spaces
std::string& trim(std::string &str)
{
  const char *spaces = " \n\r\t";
  str.erase(str.find_last_not_of(spaces) + 1);
  str.erase(0, str.find_first_not_of(spaces));
  return str;
}

unsigned char extract_mask(std::string levels)
{
  unsigned char mask = 0x0;

  std::string token;
  std::istringstream token_stream(levels);
  while (std::getline(token_stream, token, '|'))
  {
    if (token.empty())
    {
      continue;
    }
    mask |= get_log_level_type(trim(to_upper(token)));
  }

  return mask;
}

h2::Logger::LogLevelType extract_level(std::string level)
{
  h2::Logger::LogLevelType lvl;

  std::string token;
  std::istringstream token_stream(level);
  while (std::getline(token_stream, token, '|'))
  {
    if (token.empty())
    {
      continue;
    }
    //FIXME: If multiple are listed it will pick the last level in the list
    lvl = get_log_level_type(trim(to_upper(token)));
  }

  return lvl;
}

std::pair<std::string, std::string> extract_key_and_val(
  char delim, const std::string &str)
{
  auto n = str.find(delim);
  std::string key, val;
  unsigned char mask;
  if (n == std::string::npos)
  {
    return std::make_pair("", str);
  }
  else
  {
    key = str.substr(0, n);
    val = str.substr(n + 1);
  }

  return std::make_pair(trim(key), val);
}

std::unordered_map<std::string, unsigned char> get_keys_and_masks(
  std::string const& str)
{
  std::string token;
  std::istringstream token_stream(str);
  std::unordered_map<std::string, unsigned char> km{};
  while (std::getline(token_stream, token, ','))
  {
    if (token.empty())
    {
      continue;
    }
    auto kv = extract_key_and_val('=', token);

    km[kv.first] = extract_mask(kv.second);
  }
  return km;
}

std::unordered_map<std::string, h2::Logger::LogLevelType> get_keys_and_levels(
  std::string const& str)
{
  std::string token;
  std::istringstream token_stream(str);
  std::unordered_map<std::string, h2::Logger::LogLevelType> kl{};
  while (std::getline(token_stream, token, ','))
  {
    if (token.empty())
    {
      continue;
    }
    auto kv = extract_key_and_val('=', token);

    kl[kv.first] = extract_level(kv.second);
  }
  return kl;
}

}// namespace

namespace h2
{
Logger::Logger(std::string name, std::string sink_name, std::string pattern)
  : m_logger{make_logger(std::move(name), sink_name, pattern)}
{}

void Logger::set_log_level(LogLevelType level)
{
    unsigned char mask = 0x0;

    switch(level) {
        case LogLevelType::TRACE: mask |= LogLevelType::TRACE;
        case LogLevelType::DEBUG: mask |= LogLevelType::DEBUG;
        case LogLevelType::INFO: mask |= LogLevelType::INFO;
        case LogLevelType::WARN: mask |= LogLevelType::WARN;
        case LogLevelType::ERROR: mask |= LogLevelType::ERROR;
        case LogLevelType::CRITICAL: mask |= LogLevelType::CRITICAL;
          break;
        default: mask = LogLevelType::OFF;
    }

    set_mask(mask);
}

void Logger::set_mask(unsigned char mask)
{
    m_mask = mask;
}

bool Logger::should_log(LogLevelType level) const noexcept
{
    if ((m_mask & level) == level)
        return true;
    else
        return false;
}

void setup_levels(std::vector<Logger*>& loggers,
                  char const* const level_env_var)
{
  auto level_kv = get_keys_and_levels(std::getenv(level_env_var));
  auto const default_level = (level_kv.count("") ? level_kv.at("") : h2::Logger::LogLevelType::OFF);
  level_kv.erase("");

  for (auto& l : loggers)
  {
    auto const name = l->name();
    l->set_log_level(level_kv.count(name) ? level_kv.at(name) : default_level);
    level_kv.erase(name);
  }

  //FIXME: throw right error
  for (auto const& [k, _] : level_kv)
    throw std::string(k);
}

void setup_masks(std::vector<Logger*>& loggers,
                 char const* const mask_env_var)
{
  auto mask_kv = get_keys_and_masks(std::getenv(mask_env_var));
  auto const default_mask = (mask_kv.count("") ? mask_kv.at("") : h2::Logger::LogLevelType::OFF);
  mask_kv.erase("");

  for (auto& l : loggers)
  {
    auto const name = l->name();
    l->set_mask(mask_kv.count(name) ? mask_kv.at(name) : default_mask);
    mask_kv.erase(name);
  }

  //FIXME: throw right error
  for (auto const& [k, _] : mask_kv)
    throw std::string(k);
}

void setup_levels_and_masks(std::vector<Logger*>& loggers,
                            char const* const level_env_var,
                            char const* const mask_env_var)
{
  setup_levels(loggers, level_env_var);
  setup_masks(loggers, mask_env_var);
}

} // namespace h2
