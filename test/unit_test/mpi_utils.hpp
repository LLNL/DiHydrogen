////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <memory>
#include <unordered_map>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "h2/tensor/proc_grid.hpp"
#include "h2/utils/Error.hpp"


namespace internal
{
// Thrown by CommManager::get_comm if the rank is not participating.
struct NotParticipatingException {};
}

/**
 * Manages instances of communicators for testing.
 *
 * This avoids tests creating too many communicators by reusing them.
 */
class CommManager
{
public:
  CommManager()
  {
    world_size = El::mpi::COMM_WORLD.Size();
  }

  ~CommManager()
  {
    clear();
  }

  El::mpi::Comm& get_comm(int size = -1)
  {
    H2_ASSERT_ALWAYS(size <= world_size,
                     "Requested communicator size exceeds world size");

    if (size < 0)
    {
      size = world_size;
    }

    // All ranks in the world have to participate in the split, but
    // only the ones in that should be in the returned communicator
    // actually return.
    int world_rank = El::mpi::COMM_WORLD.Rank();

    if (!comms.count(size))
    {
      h2::Comm comm;
      El::mpi::Split(El::mpi::COMM_WORLD, world_rank < size, world_rank, comm);
      if (world_rank >= size)
      {
        // Need to add an entry to prevent trying to split in future
        // calls. This essentially adds MPI_COMM_NULL.
        comm.Reset();
      }
      comms.emplace(size, std::move(comm));
    }

    if (world_rank >= size)
    {
      // Not participating.
      throw internal::NotParticipatingException();
    }

    return comms[size];
  }

  void clear() { comms.clear(); }

private:
  std::unordered_map<int, h2::Comm> comms;
  int world_size;
};

/**
 * Return a communicator of the given size (-1, default, for the world).
 *
 * Throws NotParticipatingException if the caller is not present in the
 * communicator.
 */
h2::Comm& get_comm(int size = -1);
// Implemented in MPICatchMain.cpp.

/**
 * Invoke a test case with communicators of every size between the
 * specified minimum and maximum.
 *
 * This expects a callable which takes a single `h2::Comm&` as an
 * argument, which contains the actual test case to run.
 *
 * If the specified minimum requires more ranks than exist, all tests
 * will be skipped. Likewise, if the specified maximum would result in
 * some cases requiring more ranks than exist, those cases will be
 * skipped.
 *
 * A barrier (on COMM_WORLD) is performed after each communicator size.
 *
 * This should only be called inside of Catch2 test cases.
 *
 * This ensures that all ranks participate in any necessary
 * communicator creation.
 *
 * @tparam Test A callable accepting `h2::Comm&`.
 * @param t The test case to invoke.
 * @param min_size Minimum communicator size to use.
 * @param max_size Maximum communicator size to use; -1 for COMM_WORLD
 * regardless of its size.
 */
template <typename Test>
void for_comms(Test t, int min_size = 1, int max_size = -1)
{
  // Sanity check.
  H2_ASSERT_ALWAYS(max_size < 0 || min_size <= max_size,
                   "Cannot have min_size greater than max_size");

  int world_size = El::mpi::COMM_WORLD.Size();
  if (max_size < 0 || max_size > world_size)
  {
    max_size = world_size;
  }
  // Skip everything if the test requires more ranks than we have.
  if (min_size > world_size)
  {
    SKIP();
    return;
  }

  for (int i = min_size; i <= max_size; ++i)
  {
    try
    {
      h2::Comm& comm = get_comm(i);
      t(comm);
    }
    catch (const internal::NotParticipatingException&) {}
    El::mpi::Barrier(El::mpi::COMM_WORLD);
  }
}
