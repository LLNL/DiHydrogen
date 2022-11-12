////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_UTILS_LOGGING_RANK_PATTERN_HPP_INCLUDED
#define H2_UTILS_LOGGING_RANK_PATTERN_HPP_INCLUDED

#include "spdlog/pattern_formatter.h"
// #include <mpi.h>

namespace
{

class RankFlag : public spdlog::custom_flag_formatter
{
public:
    void format(const spdlog::details::log_msg&,
                const std::tm&,
                spdlog::memory_buf_t& dest) override
    {
        std::string rank = std::to_string(get_local_rank());
        // auto test = MPI_COMM_WORLD_RANK;
        dest.append(rank.data(), rank.data() + rank.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override
    {
        return spdlog::details::make_unique<RankFlag>();
    }

    /** Attempt to identify the rank from the environment. */
    int get_local_rank()
    {
        char* env = std::getenv("MV2_COMM_WORLD_RANK");
        if (!env)
            env = std::getenv("OMPI_COMM_WORLD_RANK");
        if (!env)
        {
            // Cannot determine rank
            env = "-1";
        }
        return std::atoi(env);
    }
}; // class RankFlag

} // namespace

#endif // H2_UTILS_LOGGING_RANK_PATTERN_HPP_INCLUDED
