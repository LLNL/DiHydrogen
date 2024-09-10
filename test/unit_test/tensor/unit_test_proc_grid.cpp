////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/proc_grid.hpp"
#include "utils.hpp"

#include "../mpi_utils.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

// Test that our test helpers work.

TEST_CASE("get_unique_factors works", "[dist-tensor][misc]")
{
  using ::internal::get_unique_factors;
  REQUIRE(get_unique_factors(1) == std::vector<int>{1});
  REQUIRE(get_unique_factors(2) == std::vector<int>{1, 2});
  REQUIRE(get_unique_factors(3) == std::vector<int>{1, 3});
  REQUIRE(get_unique_factors(4) == std::vector<int>{1, 2, 4});
  REQUIRE(get_unique_factors(42)
          == std::vector<int>{1, 2, 3, 6, 7, 14, 21, 42});

  REQUIRE(get_unique_factors(1, false) == std::vector<int>{});
  REQUIRE(get_unique_factors(2, false) == std::vector<int>{});
  REQUIRE(get_unique_factors(4, false) == std::vector<int>{2});
}

TEST_CASE("all_grid_shapes works", "[dist-tensor][misc]")
{
  using ::internal::all_grid_shapes;
  using ST = ShapeTuple;
  using rt = std::unordered_set<ShapeTuple>;

  REQUIRE(all_grid_shapes(1, 0, 4) == rt{ST{1}});
  REQUIRE(all_grid_shapes(2, 0, 4) == rt{ST{2}});
  REQUIRE(all_grid_shapes(4, 0, 4) == rt{ST{2, 2}, ST{4}});
  REQUIRE(all_grid_shapes(6, 0, 4) == rt{ST{2, 3}, ST{3, 2}, ST{6}});
  REQUIRE(all_grid_shapes(8, 0, 4)
          == rt{ST{2, 2, 2}, ST{2, 4}, ST{4, 2}, ST{8}});
  // Test minimum sizes.
  REQUIRE(all_grid_shapes(1, 2, 4) == rt{});
  REQUIRE(all_grid_shapes(1, 1, 4) == rt{ST{1}});
  REQUIRE(all_grid_shapes(4, 1, 4) == rt{ST{2, 2}, ST{4}});
  REQUIRE(all_grid_shapes(4, 2, 4) == rt{ST{2, 2}});
  // Test maximum sizes.
  REQUIRE(all_grid_shapes(4, 0, 1) == rt{ST{4}});
}

TEST_CASE("Processor grids can be created", "[dist-tensor][proc-grid]")
{
  for_comms([&](Comm& comm) {
    for_grid_shapes(
      [&](ShapeTuple shape) {
        REQUIRE_NOTHROW([&] {
          ProcessorGrid grid = ProcessorGrid(comm, {comm.Size()});
        }());
      },
      comm);
  });
}

TEST_CASE("Null processor grids can be created", "[dist-tensor][proc-grid]")
{
  REQUIRE_NOTHROW(ProcessorGrid());
}

TEST_CASE("Processor grids are sane", "[dist-tensor][proc-grid]")
{
  for_comms([&](Comm& comm) {
    for_grid_shapes(
      [&](ShapeTuple shape) {
        ProcessorGrid grid = ProcessorGrid(comm, shape);
        REQUIRE(grid.shape() == shape);
        REQUIRE(grid.ndim() == shape.size());
        REQUIRE(grid.size() == comm.Size());
        REQUIRE(grid.rank() == comm.Rank());
      },
      comm);
  });
}

TEST_CASE("Processor grid coordinates and ranks are sane",
          "[dist-tensor][proc-grid]")
{
  for_comms([&](Comm& comm) {
    for_grid_shapes(
      [&](ShapeTuple shape) {
        ProcessorGrid grid = ProcessorGrid(comm, shape);
        for (RankType rank = 0; rank < comm.Size(); ++rank)
        {
          auto coord = grid.coords(rank);
          REQUIRE(grid.rank(coord) == rank);
        }
      },
      comm);
  });
}

TEST_CASE("Processor grid equality works", "[dist-tensor][proc-grid]")
{
  SECTION("Empty processor grid equality")
  {
    // Empty grids always use MPI_COMM_NULL.
    ProcessorGrid grid;
    REQUIRE(grid == grid);
    REQUIRE(grid.is_identical_to(grid));
    REQUIRE_FALSE(grid != grid);
  }

  SECTION("Same-size processor grid equality")
  {
    Comm& comm = get_comm_or_skip(1);
    ProcessorGrid empty_grid;
    ProcessorGrid grid1(comm, ShapeTuple{1});

    REQUIRE_FALSE(grid1 == empty_grid);
    REQUIRE_FALSE(grid1.is_identical_to(empty_grid));
    REQUIRE(grid1 != empty_grid);
    REQUIRE(grid1 == grid1);
    REQUIRE(grid1.is_identical_to(grid1));
    REQUIRE_FALSE(grid1 != grid1);

    // New grid, even from the same comm, duplicates the underlying
    // MPI communicator.
    ProcessorGrid grid2(comm, ShapeTuple{1});
    REQUIRE_FALSE(grid1 == grid2);
    REQUIRE_FALSE(grid1.is_identical_to(grid2));
    REQUIRE_FALSE(grid2.is_identical_to(grid1));
    REQUIRE(grid1 != grid2);

    // Assignment uses the same underlying MPI comm.
    ProcessorGrid grid3 = grid1;
    REQUIRE(grid1 == grid3);
    REQUIRE(grid1.is_identical_to(grid3));
    REQUIRE(grid3.is_identical_to(grid1));
    REQUIRE_FALSE(grid1 != grid3);
  }
}

TEST_CASE("Processor grid congruence works", "[dist-tensor][proc-grid]")
{
  SECTION("Empty processor grid congruence")
  {
    ProcessorGrid grid;
    ProcessorGrid grid2;
    REQUIRE(grid.is_congruent_to(grid));
    REQUIRE(grid.is_congruent_to(grid2));
  }

  SECTION("Same-size processor grid congruence")
  {
    Comm& comm = get_comm_or_skip(1);
    ProcessorGrid empty_grid;
    ProcessorGrid grid1(comm, ShapeTuple{1});

    REQUIRE_FALSE(grid1.is_congruent_to(empty_grid));
    REQUIRE_FALSE(empty_grid.is_congruent_to(grid1));
    REQUIRE(grid1.is_congruent_to(grid1));

    ProcessorGrid grid2(comm, ShapeTuple{1});
    REQUIRE(grid1.is_congruent_to(grid2));
    REQUIRE(grid2.is_congruent_to(grid1));

    ProcessorGrid grid3 = grid1;
    REQUIRE(grid1.is_congruent_to(grid3));
    REQUIRE(grid3.is_congruent_to(grid1));
  }

  SECTION("Different-size processor grids are not congruent")
  {
    Comm& comm = get_comm_or_skip(2);
    ProcessorGrid grid1(comm, ShapeTuple{1, 2});
    ProcessorGrid grid2(comm, ShapeTuple{2, 1});

    REQUIRE_FALSE(grid1.is_congruent_to(grid2));
    REQUIRE_FALSE(grid2.is_congruent_to(grid1));
  }
}

TEST_CASE("Processor grids are printable", "[dist-tensor][proc-grid]")
{
  for_comms([&](Comm& comm) {
    for_grid_shapes(
      [&](ShapeTuple shape) {
        ProcessorGrid grid = ProcessorGrid(comm, shape);
        std::stringstream ss;
        std::stringstream shape_ss;
        print_tuple(shape_ss, shape, "(", ")", " x ");
        ss << grid;
        REQUIRE(ss.str() == std::string("Grid") + shape_ss.str());
      },
      comm);
  });
}
