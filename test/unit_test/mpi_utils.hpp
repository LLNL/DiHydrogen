////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <memory>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/Error.hpp"


namespace internal
{
// Thrown by CommManager::get_comm if the rank is not participating.
struct NotParticipatingException {};

void start_for_comms();
void end_for_comms();

}  // namespace internal

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

/** Like get_comm, but skip rather than throw if not participating. */
h2::Comm& get_comm_or_skip(int size = -1);

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

  internal::start_for_comms();

  for (int i = min_size; i <= max_size; ++i)
  {
    try
    {
      h2::Comm& comm = get_comm(i);
      t(comm);
    }
    catch (const internal::NotParticipatingException&)
    {}
    // If all the assertions in t pass, or we do not participate, we
    // reach this point. If there is a failure, we will not reach here.
    // We perform an allreduce on a single integer to determine whether
    // the test case succeeded. In the case of failure, the allreduce
    // will be joined within a Catch2 event handler.
    int test_result = 1;
    El::mpi::AllReduce(&test_result,
                       1,
                       // Because Elemental does not support LOGICAL_AND. :(
                       El::mpi::MIN,
                       El::mpi::COMM_WORLD,
                       El::SyncInfo<El::Device::CPU>{});
    if (test_result == 0)
    {
      internal::end_for_comms();  // Indicate we are done.
      FAIL(std::to_string(El::mpi::Rank())
           + ": Failure detected on another rank");
    }
  }

  internal::end_for_comms();
}

namespace internal
{

/** Return all unique factors of x in sorted order. */
template <typename T>
inline std::vector<T> get_unique_factors(T x, bool with_trivial = true)
{
  // This could be more efficient, but we shouldn't use it for large
  // inputs anyway.
  std::vector<T> factors = {};
  if (with_trivial)
  {
    factors.push_back(1);
  }
  for (T i = 2; i <= x / 2; ++i)
  {
    if (x % i == 0)
    {
      factors.push_back(i);
    }
  }
  if (x != 1 && with_trivial)
  {
    factors.push_back(x);
  }

  return factors;
}

inline std::unordered_set<h2::ShapeTuple>
all_grid_shapes(h2::ShapeTuple::type size,
                h2::ShapeTuple::type min_size,
                h2::ShapeTuple::size_type max_size)
{
  // For convenience.
  using ShapeTuple = h2::ShapeTuple;
  using type = ShapeTuple::type;
  using size_type = ShapeTuple::size_type;

  if (size == 1 && min_size <= 1)
  {
    return {ShapeTuple{1}};
  }

  std::unordered_set<ShapeTuple> shapes;
  if (min_size <= 1)
  {
    shapes.insert(ShapeTuple{size});
  }

  // Handle case where we cannot expand further.
  if (max_size == 1)
  {
    return shapes;
  }

  std::vector<type> factors = get_unique_factors(size, false);
  // Precompute the factorizations of all of the factors.
  std::unordered_map<type, std::vector<type>> factorizations;
  for (const auto& factor : factors)
  {
    factorizations[factor] = get_unique_factors(factor, false);
  }
  factorizations[size] = factors;

  // Set of shapes to expand.
  std::stack<ShapeTuple> to_expand;
  to_expand.push(ShapeTuple{size});
  while (!to_expand.empty())
  {
    ShapeTuple cur_shape = to_expand.top();
    to_expand.pop();
    for (size_type i = 0; i < cur_shape.size(); ++i)
    {
      type cur_val = cur_shape[i];
      // Sanity-check:
      H2_ASSERT_ALWAYS(factorizations.count(cur_val),
                       "No factorizations for " + std::to_string(cur_val));
      for (const auto& factor : factorizations.at(cur_val))
      {
        ShapeTuple new_shape;
        new_shape.set_size(cur_shape.size() + 1);
        // Copy preceeding elements.
        for (size_type j = 0; j < i; ++j)
        {
          new_shape[j] = cur_shape[j];
        }
        // Insert expanded value.
        new_shape[i] = factor;
        new_shape[i + 1] = cur_val / factor;
        // Copy remaining elements.
        for (size_type j = i + 1; j < cur_shape.size(); ++j)
        {
          new_shape[j + 1] = cur_shape[j];
        }
        if (new_shape.size() < max_size)
        {
          to_expand.push(new_shape);
        }
        // Add the new shape and all its permutations.
        if (new_shape.size() >= min_size)
        {
          do
          {
            shapes.insert(new_shape);
          } while (std::next_permutation(new_shape.begin(), new_shape.end()));
        }
      }
    }
  }

  return shapes;
}

}  // namespace internal

/**
 * Invoke a test case with every possible grid shape between a minimum
 * and maximum number of dimensions (both inclusive), excluding (most)
 * trivial grid shapes.
 */
template <typename Test>
void for_grid_shapes(
    Test t,
    h2::Comm& comm,
    h2::ShapeTuple::size_type min_size = 0,
    h2::ShapeTuple::size_type max_size = h2::ShapeTuple::max_size)
{
  H2_ASSERT_ALWAYS(max_size <= h2::ShapeTuple::max_size,
                   "Requested maximum grid dimensions are too large");
  H2_ASSERT_ALWAYS(max_size >= 1, "Must have at least one grid dimension");

  auto shapes = internal::all_grid_shapes(comm.Size(), min_size, max_size);

  for (const auto& shape : shapes)
  {
    t(shape);
  }
}
