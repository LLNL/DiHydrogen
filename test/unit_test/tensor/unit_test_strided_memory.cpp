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
#include "utils.hpp"

using namespace h2;


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

TEMPLATE_LIST_TEST_CASE("StridedMemory is sane",
                        "[tensor][strided_memory]",
                        AllDevList) {
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem = MemType({3, 7});
  REQUIRE(mem.data() != nullptr);
  REQUIRE(mem.const_data() != nullptr);
  REQUIRE(mem.strides() == StrideTuple{1, 3});
  REQUIRE(mem.stride(0) == 1);
  REQUIRE(mem.stride(1) == 3);
  REQUIRE(mem.shape() == ShapeTuple{3, 7});
  REQUIRE(mem.shape(0) == 3);
  REQUIRE(mem.shape(1) == 7);
}

TEMPLATE_LIST_TEST_CASE("Empty StridedMemory is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem;
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
  REQUIRE(mem.strides() == StrideTuple{});
  REQUIRE(mem.shape() == ShapeTuple{});
}

TEMPLATE_LIST_TEST_CASE("StridedMemory indexing works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem = MemType({3, 7, 2});
  // This should iterate in exactly the generalized column-major order
  // data is stored in.
  DataIndexType idx = 0;
  for (DimType k = 0; k < mem.shape(2); ++k)
  {
    for (DimType j = 0; j < mem.shape(1); ++j)
    {
      for (DimType i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(mem.get_index({i, j, k}) == idx);
        REQUIRE(mem.get_coord(idx) == SingleCoordTuple{i, j, k});
        ++idx;
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory writing works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType mem = MemType({3, 7, 2});
  DataType* buf = mem.data();
  for (std::size_t i = 0; i < product<std::size_t>(mem.shape()); ++i)
  {
    write_ele<Dev>(buf, i, static_cast<DataType>(i));
  }

  DataIndexType idx = 0;
  for (DimType k = 0; k < mem.shape(2); ++k)
  {
    for (DimType j = 0; j < mem.shape(1); ++j)
    {
      for (DimType i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(mem.get({i, j, k})) == idx);
        REQUIRE(read_ele<Dev>(mem.const_get({i, j, k})) == idx);
        ++idx;
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory views work",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType, TestType::value>;

  MemType base_mem = MemType({3, 7, 3});
  for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
  {
    write_ele<Dev>(base_mem.data(), i, static_cast<DataType>(i));
  }

  SECTION("Viewing a subtensor with all three dimensions nontrivial")
  {
    MemType mem = MemType(base_mem, {DRng(1, 3), ALL, DRng(1, 3)});
    REQUIRE(mem.strides() == StrideTuple{1, 3, 21});
    REQUIRE(mem.shape() == ShapeTuple{2, 7, 2});
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 1})));
    for (DimType k = 0; k < mem.shape(2); ++k)
    {
      for (DimType j = 0; j < mem.shape(1); ++j)
      {
        DataIndexType idx = 0;
        for (DimType i = 0; i < mem.shape(0); ++i)
        {
          REQUIRE(read_ele<Dev>(mem.get({i, j, k}))
                  == base_mem.get_index({i + 1, j, k + 1}));
          // Large enough to not be a real index.
          write_ele<Dev>(mem.get({i, j, k}), 0, static_cast<DataType>(1337));
          ++idx;
        }
      }
    }
    // Verify that writes are visible in the original StridedMemory and
    // that only the expected region was modified.
    DataIndexType idx = 0;
    for (DimType k = 0; k < base_mem.shape(2); ++k)
    {
      for (DimType j = 0; j < base_mem.shape(1); ++j)
      {
        for (DimType i = 0; i < base_mem.shape(0); ++i)
        {
          if (i >= 1 && i < 3 && k >= 1 && k < 3)
          {
            REQUIRE(read_ele<Dev>(base_mem.get({i, j, k})) == 1337);
          }
          else
          {
            REQUIRE(read_ele<Dev>(base_mem.get({i, j, k})) == idx);
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
    for (DimType k = 0; k < mem.shape(1); ++k)
    {
      for (DimType j = 0; j < mem.shape(0); ++j)
      {
        REQUIRE(read_ele<Dev>(mem.get({j, k}))
                == base_mem.get_index({1, j, k + 1}));
      }
    }
  }
}
