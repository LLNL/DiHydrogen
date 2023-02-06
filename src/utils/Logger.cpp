////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2_config.hpp"

#include "h2/utils/Logger.hpp"

#include <spdlog/cfg/env.h> // support for loading levels from the
                            // environment variable
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#if H2_HAS_MPI
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

namespace h2
{

void Logger::initialize()
{
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    auto formatter = std::make_unique<spdlog::pattern_formatter>();
    formatter->add_flag<HostnameFlag>('h');
    formatter->add_flag<MPISizeFlag>('W');
    formatter->add_flag<MPIRankFlag>('w').set_pattern(
        "[%D %H:%M %z] [%h (Rank %w/%W)] [%^%L%$] %v");

    std::vector<spdlog::sink_ptr> sinks{console_sink};

    auto logger = std::make_shared<spdlog::logger>(
        H2_LOGGER_NAME, sinks.begin(), sinks.end());
    logger->flush_on(spdlog::get_level());
    logger->set_formatter(std::move(formatter));
    spdlog::register_logger(logger);
    load_log_level();
}

void Logger::finalize()
{
    spdlog::shutdown();
}

void Logger::load_log_level()
{
    // Set the log level to "info" and mylogger to "trace":
    // SPDLOG_LEVEL=info,mylogger=trace && ./example
    spdlog::cfg::load_env_levels();
    // #define H2_LOG_ACTIVE_LEVEL SPDLOG_LEVEL
    //  or from command line:
    //  ./example SPDLOG_LEVEL=info,mylogger=trace
    // #include "spdlog/cfg/argv.h" // for loading levels from argv
    // spdlog::cfg::load_argv_levels(args, argv);
}

} // namespace h2
