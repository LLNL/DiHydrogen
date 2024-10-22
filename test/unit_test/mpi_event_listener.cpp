////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/dist_types.hpp"

// allreduce, for now
#include "mpi_utils.hpp"
#include <catch2/reporters/catch_reporter_event_listener.hpp>
#include <catch2/reporters/catch_reporter_registrars.hpp>

namespace internal
{

// These indicate when we are actually performing a blocking global
// operation.

bool in_for_comms = false;

void start_for_comms()
{
  in_for_comms = true;
}

void end_for_comms()
{
  in_for_comms = false;
}

}  // namespace internal

/**
 * Gracefully handle test case failures when not all ranks may see
 * the failure. This is intended for use with the `for_comms` utility
 * (or similar).
 */
class MPIEventListener final : public Catch::EventListenerBase
{
public:
  using Catch::EventListenerBase::EventListenerBase;

  void testCasePartialEnded(Catch::TestCaseStats const& test_case_stats,
                            uint64_t) final
  {
    if (!internal::in_for_comms)
    {
      // Not in for_comms, nothing to do.
      return;
    }

    // for_comms will join the allreduce from successful or
    // non-participating ranks.
    // We let Catch2 handle the rest of the failure as usual.
    if (!test_case_stats.totals.assertions.allOk())
    {
      int test_result = 0;
      h2_tmp::allreduce(&test_result,
                        1,
                        Al::ReductionOperator::land,
                        h2::get_comm_world(),
                        h2::ComputeStream{h2::Device::CPU});
      // Indicate we are done with the for_comms.
      internal::end_for_comms();
    }
  }
};

CATCH_REGISTER_LISTENER(MPIEventListener);
