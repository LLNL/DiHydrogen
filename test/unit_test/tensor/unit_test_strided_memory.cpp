 ////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <type_traits>

#include "h2/tensor/strided_memory.hpp"

using namespace h2;

using CPUDev_t = std::integral_constant<Device, Device::CPU>;

using DataType = float;

TEST_CASE("get_contiguous_strides", "[tensor][strided_memory]") {
  CHECK(get_contiguous_strides(ShapeTuple{}) == StrideTuple{});
  CHECK(get_contiguous_strides(ShapeTuple{1}) == StrideTuple{1});
  CHECK(get_contiguous_strides(ShapeTuple{13}) == StrideTuple{1});
  CHECK(get_contiguous_strides(ShapeTuple{13, 1}) == StrideTuple{1, 13});
  CHECK(get_contiguous_strides(ShapeTuple{13, 3}) == StrideTuple{1, 13});
  CHECK(get_contiguous_strides(ShapeTuple{13, 3, 7}) == StrideTuple{1, 13, 39});
}

TEST_CASE("are_strides_contiguous", "[tensor][strided_memory]") {
  CHECK(are_strides_contiguous(ShapeTuple{}, StrideTuple{}));
  CHECK(are_strides_contiguous(ShapeTuple{1}, StrideTuple{1}));
  CHECK(are_strides_contiguous(ShapeTuple{13}, StrideTuple{1}));
  CHECK(are_strides_contiguous(ShapeTuple{13, 1}, StrideTuple{1, 13}));
  CHECK(are_strides_contiguous(ShapeTuple{13, 3}, StrideTuple{1, 13}));
  CHECK(are_strides_contiguous(ShapeTuple{13, 3, 7}, StrideTuple{1, 13, 39}));

  CHECK_FALSE(are_strides_contiguous(ShapeTuple{1}, StrideTuple{2}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13}, StrideTuple{13}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 1}, StrideTuple{1, 2}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 1}, StrideTuple{2, 1}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 1}, StrideTuple{1, 15}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 3}, StrideTuple{1, 1}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 3}, StrideTuple{1, 3}));
  CHECK_FALSE(
      are_strides_contiguous(ShapeTuple{13, 3, 7}, StrideTuple{1, 4, 21}));
  CHECK_FALSE(are_strides_contiguous(ShapeTuple{13, 3, 7}, StrideTuple{1, 3, 32}));
}

TEST_CASE("get_contiguous_strides and are_strides_contiguous are compatible",
          "[tensor][strided_memory]")
{
  CHECK(are_strides_contiguous(ShapeTuple{},
                               get_contiguous_strides(ShapeTuple{})));
  CHECK(are_strides_contiguous(ShapeTuple{13},
                               get_contiguous_strides(ShapeTuple{13})));
  CHECK(are_strides_contiguous(ShapeTuple{13, 3, 7},
                               get_contiguous_strides(ShapeTuple{13, 3, 7})));
}

TEMPLATE_TEST_CASE("StridedMemory is sane", "[tensor][strided_memory]", CPUDev_t) {
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem = MemType({3, 7});
  REQUIRE(mem.data() != nullptr);
  REQUIRE(mem.const_data() != nullptr);
  REQUIRE(mem.strides() == StrideTuple{1, 3});
  REQUIRE(mem.shape() == ShapeTuple{3, 7});
}

TEMPLATE_TEST_CASE("Empty StridedMemory is sane",
                   "[tensor][strided_memory]",
                   CPUDev_t)
{
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem;
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
  REQUIRE(mem.strides() == StrideTuple{});
  REQUIRE(mem.shape() == ShapeTuple{});
}

TEMPLATE_TEST_CASE("StridedMemory indexing works",
                   "[tensor][strided_memory]",
                   CPUDev_t)
{
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem = MemType({3, 7, 2});
  // This should iterate in exactly the generalized column-major order
  // data is stored in.
  DataIndexType idx = 0;
  for (DimType k = 0; k < 2; ++k)
  {
    for (DimType j = 0; j < 7; ++j)
    {
      for (DimType i = 0; i < 3; ++i)
      {
        REQUIRE(mem.get_index({i, j, k}) == idx);
        REQUIRE(mem.get_coord(idx) == SingleCoordTuple{i, j, k});
        ++idx;
      }
    }
  }
}

// TODO: Support GPU devices.
TEMPLATE_TEST_CASE("StridedMemory writing works",
                   "[tensor][strided_memory]",
                   CPUDev_t)
{
  // Using DataIndexType because we check the values and floating point
  // would be a pain.
  using MemType = StridedMemory<DataIndexType, TestType::value>;

  MemType mem = MemType({3, 7, 2});
  DataIndexType* buf = mem.data();
  for (std::size_t i = 0; i < 3 * 7 * 2; ++i)
  {
    buf[i] = i;
  }

  DataIndexType idx = 0;
  for (DimType k = 0; k < 2; ++k)
  {
    for (DimType j = 0; j < 7; ++j)
    {
      for (DimType i = 0; i < 3; ++i)
      {
        REQUIRE(*mem.get({i, j, k}) == idx);
        REQUIRE(*mem.const_get({i, j, k}) == idx);
        ++idx;
      }
    }
  }
}

TEMPLATE_TEST_CASE("StridedMemory views work",
                   "[tensor][strided_memory]",
                   CPUDev_t)
{
  using MemType = StridedMemory<DataIndexType, TestType::value>;

  MemType base_mem = MemType({3, 7, 3});
  for (std::size_t i = 0; i < 3 * 7 * 3; ++i)
  {
    base_mem.data()[i] = i;
  }

  SECTION("Viewing a subtensor with all three dimensions nontrivial")
  {
    MemType mem = MemType(base_mem, {DRng(1, 3), ALL, DRng(1, 3)});
    REQUIRE(mem.strides() == StrideTuple{1, 3, 21});
    REQUIRE(mem.shape() == ShapeTuple{2, 7, 2});
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 1})));
    for (DimType k = 0; k < 2; ++k)
    {
      for (DimType j = 0; j < 7; ++j)
      {
        DataIndexType idx = 0;
        for (DimType i = 0; i < 2; ++i)
        {
          REQUIRE(*mem.get({i, j, k}) == base_mem.get_index({i+1, j, k+1}));
          *mem.get({i, j, k}) = 1337;  // Large enough to not be a real index.
          ++idx;
        }
      }
    }
    // Verify that writes are visible in the original StridedMemory and
    // that only the expected region was modified.
    DataIndexType idx = 0;
    for (DimType k = 0; k < 3; ++k)
    {
      for (DimType j = 0; j < 7; ++j)
      {
        for (DimType i = 0; i < 3; ++i)
        {
          if (i >= 1 && i < 3 && k >= 1 && k < 3)
          {
            REQUIRE(*base_mem.get({i, j, k}) == 1337);
          }
          else
          {
            REQUIRE(*base_mem.get({i, j, k}) == idx);
          }
          ++idx;
        }
      }
    }
  }
  SECTION("Viewing a subtensor with a trivial dimension")
  {
    MemType mem = MemType(base_mem, {DRng(1), ALL, DRng(1, 3)});
    REQUIRE(mem.strides() == StrideTuple{3, 21});
    REQUIRE(mem.shape() == ShapeTuple{7, 2});
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 1})));
    for (DimType k = 0; k < 2; ++k)
    {
      for (DimType j = 0; j < 7; ++j)
      {
        REQUIRE(*mem.get({j, k}) == base_mem.get_index({1, j, k+1}));
      }
    }
  }
}
