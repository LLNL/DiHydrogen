////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <mpi.h>

#include <array>
#include <iomanip>

#include <catch2/reporters/catch_reporter_cumulative_base.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace
{

void update_maxes(Catch::Counts& current,
                  std::uint64_t& current_total,
                  Catch::Counts const& in)
{
    current.passed = std::max(current.passed, in.passed);
    current.failed = std::max(current.failed, in.failed);
    current.failedButOk = std::max(current.failedButOk, in.failedButOk);
    current.skipped = std::max(current.skipped, in.skipped);
    current_total = std::max(current_total, in.total());
}

void update_maxes(Catch::Counts& current,
                  std::uint64_t& current_total,
                  Catch::Totals const& totals)
{
    update_maxes(current, current_total, totals.assertions);
    update_maxes(current, current_total, totals.testCases);
}

void write_report(std::vector<Catch::Totals> const& all_totals,
                  std::ostream& os)
{
    using array_type = std::array<std::uint64_t, 6>;
    using size_type = typename array_type::size_type;

    // Compute field widths
    enum : size_type
    {
        TOTAL = 0,
        PASS,
        FAIL,
        XFAIL,
        SKIP,
        MPISIZE
    };

    std::uint64_t max_total = 0;
    Catch::Counts max_counts;
    for (auto const& t : all_totals)
    {
        update_maxes(max_counts, max_total, t);
    }

    auto const mpi_size = all_totals.size();
    array_type fwidth = {
        max_total,
        max_counts.passed,
        max_counts.failed,
        max_counts.failedButOk,
        max_counts.skipped,
        mpi_size,
    };

    std::uint64_t width = 60; // THIS VALUE IS SPECIAL
    for (auto& x : fwidth)
    {
        x = std::to_string(x).size();
        width += x;
    }
    // Convenience and brevity
    using std::right;
    using std::setw;

    // Later reporting
    std::vector<size_t> failed_ranks;

    os << '\n' << std::string(width, '=') << '\n';
    for (size_t mpi_rank = 0; mpi_rank < mpi_size; ++mpi_rank)
    {
        auto const& t = all_totals[mpi_rank];
        if (t.assertions.failed || t.assertions.failed)
            os << "** ";
        else
            os << "   ";

        // clang-format off
        os << "RANK " << setw(fwidth[MPISIZE]) << right
           << mpi_rank << ": assertions( "
           << setw(fwidth[TOTAL]) << right << t.assertions.total() << " | "
           << setw(fwidth[PASS]) << right
           << t.assertions.passed << " pass | "
           << setw(fwidth[FAIL]) << right
           << t.assertions.failed << " fail | "
           << setw(fwidth[XFAIL]) << right
           << t.assertions.failedButOk << " xfail | "
           << setw(fwidth[SKIP]) << right
           << t.assertions.skipped << " skip )\n";
        // clang-format on

        if (t.assertions.failed > 0)
            failed_ranks.push_back(mpi_rank);
    }
    auto const num_failed_ranks = failed_ranks.size();
    if (num_failed_ranks == mpi_size)
    {
        os << "\nFAILED RANKS: ALL\n";
    }
    else if (num_failed_ranks)
    {
        auto const rwidth = std::to_string(num_failed_ranks).size();
        os << "\nFAILED RANKS:";
        for (size_t i = 0; i < num_failed_ranks; ++i)
        {
            if (i > 0 && i % 8 == 0)
                os << "\n             "; // "\nFAILED RANKS:"
            os << " " << setw(rwidth) << right << failed_ranks[i];
        }
        os << "\n";
    }
    endl(os);
    flush(os);
}
} // namespace

// Assumptions
// - All processes in MPI_COMM_WORLD participate in testing.
// - All processes in MPI_COMM_WORLD see the same command line.
// - All processes in MPI_COMM_WORLD enter the same TEST_CASE*s (this
//   includes "partial" tests cases, as Catch2 calls them).
// - Some processes in MPI_COMM_WORLD may skip some TEST_CASE*s in
//   which other processes in MPI_COMM_WORLD participate.
// - Some processes in MPI_COMM_WORLD may not participate in some
//   SECTIONs in which other processes in MPI_COMM_WORLD participate.
//
// For now, the synchronization point will be the end of the test
// case. Since every rank is assumed to be in the same TEST_CASE,
// we'll just send FILE/LINE/RETURNSTATUS for each assertion.
class MPICumulativeReporter final : public Catch::CumulativeReporterBase
{
    int rank;
    int size;

public:
    MPICumulativeReporter(Catch::ReporterConfig&& _config)
        : Catch::CumulativeReporterBase{std::move(_config)}
    {
        int flag;
        MPI_Initialized(&flag);
        if (!flag)
            throw std::runtime_error("MPI not initialized!");

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }

    ~MPICumulativeReporter() final = default;

    static std::string getDescription()
    {
        return "Report test results on Rank 0 in an MPI context.";
    }

    void testRunEnded(Catch::TestRunStats const& _testRunStats) final
    {
        Catch::CumulativeReporterBase::testRunEnded(_testRunStats);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        bool const i_am_root = (rank == 0);
        std::vector<Catch::Totals> all_totals(i_am_root ? size : 0);
        MPI_Gather(&_testRunStats.totals,
                   8,
                   MPI_UINT64_T,
                   all_totals.data(),
                   8,
                   MPI_UINT64_T,
                   0,
                   MPI_COMM_WORLD);

        if (i_am_root)
            write_report(all_totals, m_stream);
    }

    void testRunEndedCumulative() final {}
};

CATCH_REGISTER_REPORTER("mpicumulative", MPICumulativeReporter);
