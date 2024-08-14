////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <type_traits>

#include "h2/tensor/strided_memory.hpp"
#include "h2/utils/typename.hpp"
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

TEST_CASE("get_extent_from_strides works", "[tensor][strided_memor]")
{
  // Contiguous:
  REQUIRE(get_extent_from_strides(ShapeTuple{}, StrideTuple{}) == 0);
  REQUIRE(get_extent_from_strides(ShapeTuple{1}, StrideTuple{1}) == 1);
  REQUIRE(get_extent_from_strides(ShapeTuple{13}, StrideTuple{1}) == 13);
  REQUIRE(get_extent_from_strides(ShapeTuple{13, 1}, StrideTuple{1, 13}) == 13);
  REQUIRE(get_extent_from_strides(ShapeTuple{13, 3, 7}, StrideTuple{1, 13, 39})
          == 273);

  // Non-contiguous:
  REQUIRE(get_extent_from_strides(ShapeTuple{1}, StrideTuple{2}) == 1);
  REQUIRE(get_extent_from_strides(ShapeTuple{13}, StrideTuple{2}) == 25);
  REQUIRE(get_extent_from_strides(ShapeTuple{13, 3}, StrideTuple{1, 15}) == 43);
  REQUIRE(get_extent_from_strides(ShapeTuple{13, 3, 7}, StrideTuple{2, 13, 50}) == 351);
}

TEMPLATE_LIST_TEST_CASE("StridedMemory is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{3, 7}, false, ComputeStream{Dev});
  REQUIRE(mem.size() == 21);
  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() != nullptr);
  REQUIRE(mem.const_data() != nullptr);
  REQUIRE(mem.strides() == StrideTuple{1, 3});
  REQUIRE(mem.stride(0) == 1);
  REQUIRE(mem.stride(1) == 3);
  REQUIRE(mem.shape() == ShapeTuple{3, 7});
  REQUIRE(mem.shape(0) == 3);
  REQUIRE(mem.shape(1) == 7);
  REQUIRE_FALSE(mem.is_lazy());

  SECTION("Ensure then release works")
  {
    DataType* orig_data = mem.data();
    mem.ensure();
    REQUIRE(mem.size() == 21);
    REQUIRE(mem.data() == orig_data);
    REQUIRE(mem.const_data() == orig_data);
    mem.ensure();
    REQUIRE(mem.size() == 21);
    REQUIRE(mem.data() == orig_data);
    REQUIRE(mem.const_data() == orig_data);
    mem.release();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.release();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.size() == 21);
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
  }
  SECTION("Release then ensure works")
  {
    mem.release();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.size() == 21);
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("Empty StridedMemory is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem{Dev, false, ComputeStream{Dev}};
  REQUIRE(mem.size() == 0);
  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
  REQUIRE(mem.strides() == StrideTuple{});
  REQUIRE(mem.shape() == ShapeTuple{});
  REQUIRE_FALSE(mem.is_lazy());

  SECTION("Ensure then release works")
  {
    mem.ensure();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.release();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
  }
  SECTION("Release then ensure works")
  {
    mem.release();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.size() == 0);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory with empty shape is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{}, false, ComputeStream{Dev});
  REQUIRE(mem.size() == 0);
  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
  REQUIRE(mem.strides() == StrideTuple{});
  REQUIRE(mem.shape() == ShapeTuple{});
  REQUIRE_FALSE(mem.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("StridedMemory with zero in shape is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{7, 0}, false, ComputeStream{Dev});
  REQUIRE(mem.size() == 0);
  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
  REQUIRE(mem.shape() == ShapeTuple{7, 0});
  REQUIRE_FALSE(mem.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("StridedMemory indexing works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{3, 7, 2}, false, ComputeStream{Dev});
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
        REQUIRE(mem.get_coord(idx) == ScalarIndexTuple{i, j, k});
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
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{3, 7, 2}, false, ComputeStream{Dev});
  DataType* buf = mem.data();
  for (std::size_t i = 0; i < product<std::size_t>(mem.shape()); ++i)
  {
    write_ele<Dev>(buf, i, static_cast<DataType>(i), mem.get_stream());
  }

  DataIndexType idx = 0;
  for (DimType k = 0; k < mem.shape(2); ++k)
  {
    for (DimType j = 0; j < mem.shape(1); ++j)
    {
      for (DimType i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(mem.get({i, j, k}), mem.get_stream()) == idx);
        REQUIRE(read_ele<Dev>(mem.const_get({i, j, k}), mem.get_stream())
                == idx);
        ++idx;
      }
    }
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory with non-contiguous strides is sane",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem(Dev,
              ShapeTuple{3, 7, 2},
              StrideTuple{2, 4, 21},
              false,
              ComputeStream{Dev});

  REQUIRE(mem.size() == 50);
  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() != nullptr);
  REQUIRE(mem.const_data() != nullptr);
  REQUIRE(mem.strides() == StrideTuple{2, 4, 21});
  REQUIRE(mem.shape() == ShapeTuple{3, 7, 2});
  REQUIRE_FALSE(mem.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("StridedMemory views work",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType base_mem =
      MemType(Dev, ShapeTuple{3, 7, 3}, false, ComputeStream{Dev});
  for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
  {
    write_ele<Dev>(
      base_mem.data(), i, static_cast<DataType>(i), base_mem.get_stream());
  }

  SECTION("Viewing a subtensor with all three dimensions nontrivial")
  {
    MemType mem = MemType(base_mem, {IRng(1, 3), ALL, IRng(1, 3)});
    REQUIRE(mem.strides() == StrideTuple{1, 3, 21});
    REQUIRE(mem.shape() == ShapeTuple{2, 7, 2});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 1})));
    REQUIRE_FALSE(mem.is_lazy());
    for (DimType k = 0; k < mem.shape(2); ++k)
    {
      for (DimType j = 0; j < mem.shape(1); ++j)
      {
        for (DimType i = 0; i < mem.shape(0); ++i)
        {
          REQUIRE(read_ele<Dev>(mem.get({i, j, k}), mem.get_stream())
                  == base_mem.get_index({i + 1, j, k + 1}));
          // Large enough to not be a real index.
          write_ele<Dev>(mem.get({i, j, k}),
                         0,
                         static_cast<DataType>(1337),
                         mem.get_stream());
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
            REQUIRE(read_ele<Dev>(
                      base_mem.get({i, j, k}), base_mem.get_stream()) == 1337);
          }
          else
          {
            REQUIRE(read_ele<Dev>(
                      base_mem.get({i, j, k}), base_mem.get_stream()) == idx);
          }
          ++idx;
        }
      }
    }
  }
  SECTION("Viewing a subtensor with a scalar dimension works")
  {
    MemType mem = MemType(base_mem, {IRng(1), ALL, IRng(1, 3)});
    REQUIRE(mem.strides() == StrideTuple{3, 21});
    REQUIRE(mem.shape() == ShapeTuple{7, 2});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 1})));
    REQUIRE_FALSE(mem.is_lazy());
    for (DimType k = 0; k < mem.shape(1); ++k)
    {
      for (DimType j = 0; j < mem.shape(0); ++j)
      {
        REQUIRE(read_ele<Dev>(mem.get({j, k}), mem.get_stream())
                == base_mem.get_index({1, j, k + 1}));
      }
    }
  }
  SECTION("Viewing a subtensor with a range of length 1 works")
  {
    MemType mem = MemType(base_mem, {IRng(0, 2), IRng(1, 2), ALL});
    REQUIRE(mem.strides() == StrideTuple{1, 3, 21});
    REQUIRE(mem.shape() == ShapeTuple{2, 1, 3});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({0, 1, 0})));
    REQUIRE_FALSE(mem.is_lazy());
    for (DimType k = 0; k < mem.shape(2); ++k)
    {
      for (DimType i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(mem.get({i, 0, k}), mem.get_stream())
                              == base_mem.get_index({i, 1, k}));
      }
    }
  }
  SECTION("Viewing with all scalar coordinates works")
  {
    MemType mem = MemType(base_mem, {IRng(1), IRng(0), IRng(0)});
    REQUIRE(mem.strides() == StrideTuple{1});
    REQUIRE(mem.shape() == ShapeTuple{1});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == (base_mem.data() + base_mem.get_index({1, 0, 0})));
    REQUIRE_FALSE(mem.is_lazy());
    REQUIRE(read_ele<Dev>(mem.get({0}), mem.get_stream())
            == base_mem.get_index({1, 0, 0}));
  }
  SECTION("Views with totally empty coordinates work")
  {
    MemType mem = MemType(base_mem, IndexRangeTuple{});
    REQUIRE(mem.strides() == StrideTuple{});
    REQUIRE(mem.shape() == ShapeTuple{});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == nullptr);
  }
  SECTION("Views with empty coordinates work")
  {
    MemType mem = MemType(base_mem, {IRng(0, 1), IRng(), ALL});
    REQUIRE(mem.strides() == StrideTuple{});
    REQUIRE(mem.shape() == ShapeTuple{});
    REQUIRE(mem.size() == base_mem.size());
    REQUIRE(mem.data() == nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory views of subviews work",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType base_mem =
    MemType(Dev, ShapeTuple{4, 6}, false, ComputeStream{Dev});
  for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
  {
    write_ele<Dev>(
      base_mem.data(), i, static_cast<DataType>(i), base_mem.get_stream());
  }

  auto view_orig = MemType(base_mem, {ALL, IRng(1, 3)});
  auto view = MemType(view_orig, {ALL, ALL});

  REQUIRE(view.shape() == view_orig.shape());
  REQUIRE(view.strides() == view_orig.strides());
  REQUIRE(view.size() == view_orig.size());
  REQUIRE(view.data() == view_orig.data());
}

TEMPLATE_LIST_TEST_CASE("Lazy StridedMemory works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, ShapeTuple{3, 7}, true, ComputeStream{Dev});
  REQUIRE(mem.is_lazy());
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);

  SECTION("Ensure and release work")
  {
    mem.ensure();
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
    mem.ensure();  // Test calling ensure multiple times.
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
    mem.release();
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.release();
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
  }

  SECTION("Buffer recovery works")
  {
    mem.ensure();
    MemType mem2 = mem;
    REQUIRE(mem.data() == mem2.data());
    REQUIRE(mem.const_data() == mem2.const_data());
    mem.release();
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.data() == mem2.data());
    REQUIRE(mem.const_data() == mem2.const_data());
    mem.release();
    mem.ensure(false);
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.data() != mem2.data());
    REQUIRE(mem.const_data() != nullptr);
    REQUIRE(mem.const_data() != mem2.const_data());
    mem.release();
    mem2.release();
    mem.ensure();
    mem2.ensure();
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
    REQUIRE(mem2.data() != nullptr);
    REQUIRE(mem2.const_data() != nullptr);
    REQUIRE(mem.data() != mem2.data());
    REQUIRE(mem.const_data() != mem2.const_data());
  }

  SECTION("Viewing lazy strided memory works")
  {
    MemType mem2 = mem;
    REQUIRE(mem2.is_lazy());
    REQUIRE(mem2.data() == nullptr);
    REQUIRE(mem2.const_data() == nullptr);

    SECTION("Ensure from view 1 works")
    {
      mem.ensure();
      REQUIRE(mem.data() != nullptr);
      REQUIRE(mem.const_data() != nullptr);
      REQUIRE(mem.data() == mem2.data());
      REQUIRE(mem.const_data() == mem2.const_data());
      mem.release();
      REQUIRE(mem.data() == nullptr);
      REQUIRE(mem.const_data() == nullptr);
      REQUIRE(mem2.data() != nullptr);
      REQUIRE(mem2.const_data() != nullptr);
    }
    SECTION("Ensure from view 2 works")
    {
      mem2.ensure();
      REQUIRE(mem2.data() != nullptr);
      REQUIRE(mem2.const_data() != nullptr);
      REQUIRE(mem.data() == mem2.data());
      REQUIRE(mem.const_data() == mem2.const_data());
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Empty lazy StridedMemory works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  MemType mem = MemType(Dev, true, ComputeStream{Dev});
  REQUIRE(mem.is_lazy());
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);

  mem.ensure();
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);

  mem.release();
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);

  mem.ensure();
  REQUIRE(mem.data() == nullptr);
  REQUIRE(mem.const_data() == nullptr);
}

TEMPLATE_LIST_TEST_CASE("StridedMemory with external buffers works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  DataType test_data[] = {0, 0, 0, 0};
  MemType mem = MemType(
      Dev, test_data, ShapeTuple{2, 2}, StrideTuple{1, 2}, ComputeStream{Dev});

  REQUIRE(mem.get_device() == Dev);
  REQUIRE(mem.data() == test_data);
  REQUIRE(mem.const_data() == test_data);
  REQUIRE(mem.shape() == ShapeTuple{2, 2});
  REQUIRE(mem.strides() == StrideTuple{1, 2});

  SECTION("Ensure and release works")
  {
    mem.ensure();
    REQUIRE(mem.data() == test_data);
    REQUIRE(mem.const_data() == test_data);
    mem.release();
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
  }

  SECTION("Views work")
  {
    MemType mem2 = mem;
    REQUIRE(mem.data() == mem2.data());
    REQUIRE(mem.const_data() == mem2.const_data());
    mem.release();
    REQUIRE(mem.data() == nullptr);
    REQUIRE(mem.const_data() == nullptr);
    mem.ensure();
    REQUIRE(mem.data() == mem2.data());
    REQUIRE(mem.const_data() == mem2.const_data());
    mem.release();
    mem.ensure(false);
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.data() != mem2.data());
    REQUIRE(mem.const_data() != nullptr);
    REQUIRE(mem.const_data() != mem2.const_data());
    mem.release();
    mem2.release();
    mem.ensure();
    mem2.ensure();
    REQUIRE(mem.data() != nullptr);
    REQUIRE(mem.const_data() != nullptr);
    REQUIRE(mem2.data() != nullptr);
    REQUIRE(mem2.const_data() != nullptr);
    REQUIRE(mem.data() != mem2.data());
    REQUIRE(mem.const_data() != mem2.const_data());
  }
}

#ifdef H2_TEST_WITH_GPU
// Only makes sense when we have GPU support.

TEMPLATE_LIST_TEST_CASE("StridedMemory views across devices work",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device SrcDev = TestType::value;
  constexpr Device DstDev = (SrcDev == Device::CPU) ? Device::GPU : Device::CPU;
  using MemType = StridedMemory<DataType>;

  ComputeStream src_stream{SrcDev};
  ComputeStream dst_stream{DstDev};
  MemType mem(SrcDev, {3, 5}, false, src_stream);

  // Note:
  // Do not attempt to access the data since we don't check if it is
  // actually safe, just that the pointers are the same.
  MemType mem_view(mem, DstDev, dst_stream);

  REQUIRE(mem_view.get_device() == DstDev);
  REQUIRE(mem_view.data() == mem.data());
  REQUIRE(mem_view.const_data() == mem.const_data());
  REQUIRE(mem_view.shape() == mem.shape());
  REQUIRE(mem_view.strides() == mem.strides());
  REQUIRE(mem_view.is_lazy() == mem.is_lazy());
  REQUIRE(mem_view.get_stream() == dst_stream);
}

#endif  // H2_TEST_WITH_GPU

TEMPLATE_LIST_TEST_CASE("Cloning StridedMemory works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  SECTION("Cloning an empty StridedMemory")
  {
    MemType mem(Dev, false, ComputeStream{Dev});
    MemType clone = mem.clone();
    REQUIRE(clone.data() == nullptr);
    REQUIRE(clone.size() == mem.size());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == StrideTuple{});
    REQUIRE(clone.shape() == ShapeTuple{});
    REQUIRE(clone.is_lazy() == mem.is_lazy());
  }

  SECTION("Cloning a regular StridedMemory")
  {
    MemType mem(Dev, ShapeTuple{3, 7}, false, ComputeStream{Dev});
    for (std::size_t i = 0; i < product<std::size_t>(mem.shape()); ++i)
    {
      write_ele<Dev>(mem.data(), i, static_cast<DataType>(i), mem.get_stream());
    }
    MemType clone = mem.clone();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.size() == mem.size());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
    for (std::size_t i = 0; i < product<std::size_t>(clone.shape()); ++i)
    {
      REQUIRE(read_ele<Dev>(clone.data(), i, clone.get_stream()) == i);
    }
  }

  SECTION("Cloning a lazy StridedMemory after ensure")
  {
    MemType mem(Dev, ShapeTuple{3, 7}, true, ComputeStream{Dev});
    mem.ensure();
    MemType clone = mem.clone();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.size() == mem.size());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
  }

  SECTION("Cloning a lazy StridedMemory before ensure")
  {
    MemType mem(Dev, ShapeTuple{3, 7}, true, ComputeStream{Dev});
    MemType clone = mem.clone();
    REQUIRE(clone.is_lazy());
    REQUIRE(clone.data() == nullptr);
    clone.ensure();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(mem.data() == nullptr);
    REQUIRE(clone.size() == mem.size());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
  }

  SECTION("Cloning a contiguous StridedMemory view")
  {
    MemType base_mem(Dev, ShapeTuple{3, 7, 3}, false, ComputeStream{Dev});
    for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
    {
      write_ele<Dev>(
        base_mem.data(), i, static_cast<DataType>(i), base_mem.get_stream());
    }
    MemType mem(base_mem, {ALL, ALL, ALL});
    MemType clone = mem.clone();
    clone.get_stream().wait_for_this();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.size() == mem.size());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
    for (std::size_t i = 0; i < product<std::size_t>(clone.shape()); ++i)
    {
      REQUIRE(read_ele<Dev>(clone.data(), i, clone.get_stream()) == i);
    }
  }

  // Not checking size() because the buffer may change.
  SECTION("Cloning a contiguous, offset StridedMemory view")
  {
    MemType base_mem(Dev, ShapeTuple{3, 7, 3}, false, ComputeStream{Dev});
    for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
    {
      write_ele<Dev>(
        base_mem.data(), i, static_cast<DataType>(i), base_mem.get_stream());
    }
    MemType mem(base_mem, {ALL, ALL, IRng(1)});
    MemType clone = mem.clone();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
    for (DimType j = 0; j < clone.shape(1); ++j)
    {
      for (DimType i = 0; i < clone.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(clone.get({i, j}), clone.get_stream())
                == base_mem.get_index({i, j, 1}));
      }
    }
  }

  SECTION("Cloning a non-contiguous StridedMemory view")
  {
    MemType base_mem(Dev, ShapeTuple{3, 7, 3}, false, ComputeStream{Dev});
    for (std::size_t i = 0; i < product<std::size_t>(base_mem.shape()); ++i)
    {
      write_ele<Dev>(
        base_mem.data(), i, static_cast<DataType>(i), base_mem.get_stream());
    }
    MemType mem(base_mem, {IRng(1), ALL, ALL});
    MemType clone = mem.clone();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
    for (DimType k = 0; k < clone.shape(1); ++k)
    {
      for (DimType j = 0; j < clone.shape(0); ++j)
      {
        REQUIRE(read_ele<Dev>(clone.get({j, k}), clone.get_stream())
                == base_mem.get_index({1, j, k}));
      }
    }
  }

  SECTION("Cloning a StridedMemory wrapping an external buffer")
  {
    constexpr std::size_t buf_size = 4 * 6;
    DeviceBuf<DataType, Dev> buf(buf_size);
    for (std::size_t i = 0; i < buf_size; ++i)
    {
      write_ele<Dev>(buf.buf, i, static_cast<DataType>(i), ComputeStream{Dev});
    }

    MemType mem(
        Dev, buf.buf, ShapeTuple{4, 6}, StrideTuple{1, 4}, ComputeStream{Dev});
    MemType clone = mem.clone();
    REQUIRE(clone.data() != nullptr);
    REQUIRE(clone.data() != mem.data());
    REQUIRE(clone.get_device() == Dev);
    REQUIRE(clone.strides() == mem.strides());
    REQUIRE(clone.shape() == mem.shape());
    REQUIRE(clone.is_lazy() == mem.is_lazy());
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory get works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  SECTION("Get from contiguous StridedMemory")
  {
    MemType mem{Dev, {2, 4}, false, ComputeStream{Dev}};
    for (std::size_t i = 0; i < product<std::size_t>(mem.shape()); ++i)
    {
      write_ele<Dev>(mem.data(), i, static_cast<DataType>(i), mem.get_stream());
    }

    std::size_t v = 0;
    for (typename ShapeTuple::type j = 0; j < mem.shape(1); ++j)
    {
      for (typename ShapeTuple::type i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(mem.get({i, j}), mem.get_stream())
                == static_cast<DataType>(v));
        ++v;
      }
    }
  }

  SECTION("Get from non-contiguous StridedMemory")
  {
    MemType mem{Dev, {2, 4}, {1, 4}, false, ComputeStream{Dev}};
    std::size_t v = 0;
    for (typename ShapeTuple::type j = 0; j < mem.shape(1); ++j)
    {
      for (typename ShapeTuple::type i = 0; i < mem.shape(0); ++i)
      {
        write_ele<Dev>(
            mem.data(), i + 4 * j, static_cast<DataType>(v), mem.get_stream());
        ++v;
      }
    }

    v = 0;
    for (typename ShapeTuple::type j = 0; j < mem.shape(1); ++j)
    {
      for (typename ShapeTuple::type i = 0; i < mem.shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(mem.get({i, j}), mem.get_stream())
                == static_cast<DataType>(v));
        ++v;
      }
    }
  }

  SECTION("Get from a non-contiguous StridedMemory view")
  {
    MemType mem{Dev, {2, 4}, false, ComputeStream{Dev}};
    for (std::size_t i = 0; i < product<std::size_t>(mem.shape()); ++i)
    {
      write_ele<Dev>(mem.data(), i, static_cast<DataType>(i), mem.get_stream());
    }

    MemType mem_view{mem, {ALL, IRng{1, 3}}};
    REQUIRE(read_ele<Dev>(mem_view.get({0, 0}), mem_view.get_stream()) == 2);
    REQUIRE(read_ele<Dev>(mem_view.get({0, 1}), mem_view.get_stream()) == 4);
    REQUIRE(read_ele<Dev>(mem_view.get({1, 0}), mem_view.get_stream()) == 3);
    REQUIRE(read_ele<Dev>(mem_view.get({1, 1}), mem_view.get_stream()) == 5);
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory get/set stream works",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  ComputeStream stream1 = create_new_compute_stream<Dev>();
  ComputeStream stream2 = create_new_compute_stream<Dev>();

  SECTION("No raw set")
  {
    MemType mem(Dev, {3, 5}, false, stream1);
    REQUIRE(mem.get_stream() == stream1);
    mem.set_stream(stream2, false);
    REQUIRE(mem.get_stream() == stream2);
  }

  SECTION("Raw set")
  {
    MemType mem(Dev, {3, 5}, false, stream1);
    REQUIRE(mem.get_stream() == stream1);
    mem.set_stream(stream2, true);
    REQUIRE(mem.get_stream() == stream2);
  }

  SECTION("No raw set with empty raw buffer")
  {
    MemType mem(Dev, false, stream1);
    REQUIRE(mem.get_stream() == stream1);
    mem.set_stream(stream2, false);
    REQUIRE(mem.get_stream() == stream2);
  }

  SECTION("Raw set with empty raw buffer")
  {
    MemType mem(Dev, false, stream1);
    REQUIRE(mem.get_stream() == stream1);
    mem.set_stream(stream2, true);
    REQUIRE(mem.get_stream() == stream2);
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory is printable",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  std::stringstream dev_ss;
  dev_ss << TestType::value;

  std::stringstream ss;

  SECTION("Lazy")
  {
    MemType mem(Dev, {3, 5}, true, ComputeStream{Dev});
    ss << mem;

    REQUIRE_THAT(ss.str(),
                 Catch::Matchers::StartsWith(std::string("StridedMemory<")
                                             + TypeName<DataType>() + ", "
                                             + dev_ss.str() + ">(lazy"));
    REQUIRE_THAT(ss.str(), Catch::Matchers::EndsWith("{3, 5})"));
  }

  SECTION("Unlazy")
  {
    MemType mem(Dev, {3, 5}, false, ComputeStream{Dev});
    ss << mem;

    REQUIRE_THAT(ss.str(),
                 Catch::Matchers::StartsWith(std::string("StridedMemory<")
                                             + TypeName<DataType>() + ", "
                                             + dev_ss.str() + ">(not lazy"));
    REQUIRE_THAT(ss.str(), Catch::Matchers::EndsWith("{3, 5})"));
  }
}

TEMPLATE_LIST_TEST_CASE("StridedMemory contents print",
                        "[tensor][strided_memory]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using MemType = StridedMemory<DataType>;

  SECTION("Printing empty StridedBuffers works")
  {
    MemType mem{Dev, false, ComputeStream{Dev}};
    std::stringstream ss;
    strided_memory_contents(ss, mem);
    REQUIRE(ss.str() == "");
  }

  SECTION("Printing single-element StridedBuffers works")
  {
    MemType mem{Dev, ShapeTuple{1}, false, ComputeStream{Dev}};
    write_ele<Dev>(mem.data(), 0, static_cast<DataType>(1), mem.get_stream());
    std::stringstream ss;
    strided_memory_contents(ss, mem);
    REQUIRE(ss.str() == "1");
  }

  SECTION("Printing contiguous StridedBuffers works")
  {
    MemType mem{Dev, ShapeTuple{2, 3}, false, ComputeStream{Dev}};
    std::stringstream expected_ss;
    std::size_t size = product<std::size_t>(mem.shape());
    for (std::size_t i = 0; i < size; ++i)
    {
      write_ele<Dev>(mem.data(), i, static_cast<DataType>(i), mem.get_stream());
      expected_ss << static_cast<DataType>(i);
      if (i != size - 1)
      {
        expected_ss << ", ";
      }
    }

    std::stringstream ss;
    strided_memory_contents(ss, mem);
    REQUIRE(ss.str() == expected_ss.str());
  }

  SECTION("Printing non-contiguous StridedBuffers works")
  {
    MemType mem{
        Dev, ShapeTuple{2, 3}, StrideTuple{2, 4}, false, ComputeStream{Dev}};
    std::stringstream expected_ss;
    DataIndexType size = product<DataIndexType>(mem.shape());
    DataIndexType v = 0;
    for (DimType j = 0; j < mem.shape(1); ++j)
    {
      for (DimType i = 0; i < mem.shape(0); ++i)
      {
        write_ele<Dev>(
            mem.get({i, j}), 0, static_cast<DataType>(v), mem.get_stream());
        expected_ss << static_cast<DataType>(v);
        if (v != size - 1)
        {
          expected_ss << ", ";
        }
        ++v;
      }
    }

    std::stringstream ss;
    strided_memory_contents(ss, mem);
    REQUIRE(ss.str() == expected_ss.str());
  }
}
