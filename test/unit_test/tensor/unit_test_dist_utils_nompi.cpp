////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/dist_utils.hpp"
#include "utils.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace h2;

TEST_CASE("Dimension local size works", "[dist-tensor][utils]")
{
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Block>(4, 2, 1, false)
          == 2);
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Block>(4, 3, 0, false)
          == 2);
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Block>(4, 3, 1, false)
          == 1);
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Block>(4, 5, 0, false)
          == 1);
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Block>(4, 5, 4, false)
          == 0);

  REQUIRE(
    h2::internal::get_dim_local_size<Distribution::Replicated>(4, 2, 0, false)
    == 4);
  REQUIRE(
    h2::internal::get_dim_local_size<Distribution::Replicated>(4, 2, 1, false)
    == 4);
  REQUIRE(
    h2::internal::get_dim_local_size<Distribution::Replicated>(4, 5, 4, false)
    == 4);

  REQUIRE(h2::internal::get_dim_local_size<Distribution::Single>(4, 2, 0, true)
          == 4);
  REQUIRE(h2::internal::get_dim_local_size<Distribution::Single>(4, 2, 1, false)
          == 0);
}

TEST_CASE("Dimension global coordinates works", "[dist-tensor][utils]")
{
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 2, 0, false)
    == IRng(0, 2));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 2, 1, false)
    == IRng(2, 4));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 3, 0, false)
    == IRng(0, 2));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 3, 1, false)
    == IRng(2, 3));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 3, 2, false)
    == IRng(3, 4));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 5, 0, false)
    == IRng(0, 1));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Block>(4, 5, 4, false)
      .is_empty());

  REQUIRE(h2::internal::get_dim_global_indices<Distribution::Replicated>(
            4, 2, 0, false)
          == IRng(0, 4));
  REQUIRE(h2::internal::get_dim_global_indices<Distribution::Replicated>(
            4, 2, 1, false)
          == IRng(0, 4));
  REQUIRE(h2::internal::get_dim_global_indices<Distribution::Replicated>(
            4, 5, 4, false)
          == IRng(0, 4));

  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Single>(4, 2, 0, true)
    == IRng(0, 4));
  REQUIRE(
    h2::internal::get_dim_global_indices<Distribution::Single>(4, 2, 0, false)
      .is_empty());
}

TEST_CASE("Dimension global-to-local index works", "[dist-tensor][utils]")
{
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 2, 1)
          == 1);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 2, 3)
          == 1);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 3, 0)
          == 0);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 3, 1)
          == 1);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 3, 2)
          == 0);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 3, 3)
          == 0);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Block>(4, 5, 1)
          == 0);

  REQUIRE(
    h2::internal::dim_global2local_index<Distribution::Replicated>(4, 2, 0)
    == 0);
  REQUIRE(
    h2::internal::dim_global2local_index<Distribution::Replicated>(4, 2, 1)
    == 1);
  REQUIRE(
    h2::internal::dim_global2local_index<Distribution::Replicated>(4, 5, 1)
    == 1);

  REQUIRE(h2::internal::dim_global2local_index<Distribution::Single>(4, 2, 0)
          == 0);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Single>(4, 2, 1)
          == 1);
  REQUIRE(h2::internal::dim_global2local_index<Distribution::Single>(4, 5, 1)
          == 1);
}

TEST_CASE("Dimension global index-to-rank works", "[dist-tensor][utils]")
{
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 2, 1) == 0);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 2, 2) == 1);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 3, 0) == 0);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 3, 1) == 0);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 3, 2) == 1);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 3, 3) == 2);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 5, 0) == 0);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Block>(4, 5, 3) == 3);

  REQUIRE(h2::internal::dim_global2rank<Distribution::Replicated>(4, 2, 1)
          == 0);

  REQUIRE(h2::internal::dim_global2rank<Distribution::Single>(4, 2, 1) == 0);
  REQUIRE(h2::internal::dim_global2rank<Distribution::Single>(4, 2, 3) == 0);
}

TEST_CASE("Dimension local-to-global index works", "[dist-tensor][utils]")
{
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 2, 0, 0)
          == 0);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 2, 0, 1)
          == 1);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 2, 1, 0)
          == 2);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 2, 1, 1)
          == 3);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 3, 0, 0)
          == 0);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 3, 0, 1)
          == 1);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 3, 1, 0)
          == 2);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 3, 2, 0)
          == 3);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 5, 0, 0)
          == 0);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Block>(4, 5, 3, 0)
          == 3);

  REQUIRE(
    h2::internal::dim_local2global_index<Distribution::Replicated>(4, 2, 0, 1)
    == 1);
  REQUIRE(
    h2::internal::dim_local2global_index<Distribution::Replicated>(4, 2, 1, 1)
    == 1);

  REQUIRE(h2::internal::dim_local2global_index<Distribution::Single>(4, 2, 0, 1)
          == 1);
  REQUIRE(h2::internal::dim_local2global_index<Distribution::Single>(4, 2, 0, 3)
          == 3);
}
