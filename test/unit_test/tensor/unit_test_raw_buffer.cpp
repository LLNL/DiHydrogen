////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <type_traits>

#include "h2/tensor/raw_buffer.hpp"
#include "utils.hpp"

using namespace h2;


TEMPLATE_LIST_TEST_CASE("Raw buffers are sane",
                        "[tensor][raw_buffer]",
                        AllDevList)
{
  using BufType = RawBuffer<DataType, TestType::value>;
  constexpr std::size_t buf_size = 32;

  BufType buf = BufType(buf_size);

  REQUIRE(buf.size() == buf_size);
  REQUIRE(buf.data() != nullptr);
  REQUIRE(buf.const_data() != nullptr);

  DataType* orig_buf = buf.data();

  SECTION("Ensure then release works") {
    buf.ensure();
    REQUIRE(buf.size() == buf_size);
    REQUIRE(buf.data() == orig_buf);
    REQUIRE(buf.const_data() == orig_buf);

    buf.release();
    REQUIRE(buf.size() == buf_size);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);
  }
  SECTION("Release then ensure works") {
    buf.release();
    REQUIRE(buf.size() == buf_size);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);

    buf.ensure();
    REQUIRE(buf.size() == buf_size);
    // No guarantee it's the same or different underlying pointer.
    REQUIRE(buf.data() != nullptr);
    REQUIRE(buf.const_data() != nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("Empty raw buffers are sane",
                        "[tensor][raw_buffer]",
                        AllDevList)
{
  using BufType = RawBuffer<DataType, TestType::value>;

  BufType buf;

  REQUIRE(buf.size() == 0);
  REQUIRE(buf.data() == nullptr);
  REQUIRE(buf.const_data() == nullptr);

  SECTION("Ensure then release works") {
    buf.ensure();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);

    buf.release();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);
  }
  SECTION("Release then ensure works") {
    buf.release();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);

    buf.ensure();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("Raw buffer with explicit size 0 is sane",
                        "[tensor][raw_buffer]",
                        AllDevList)
{
  using BufType = RawBuffer<DataType, TestType::value>;

  BufType buf(0);

  REQUIRE(buf.size() == 0);
  REQUIRE(buf.data() == nullptr);
  REQUIRE(buf.const_data() == nullptr);

  SECTION("Ensure then release works") {
    buf.ensure();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);

    buf.release();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);
  }
  SECTION("Release then ensure works") {
    buf.release();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);

    buf.ensure();
    REQUIRE(buf.size() == 0);
    REQUIRE(buf.data() == nullptr);
    REQUIRE(buf.const_data() == nullptr);
  }
}

TEMPLATE_LIST_TEST_CASE("Raw buffer with external memory is sane",
                        "[tensor][raw_buffer]",
                        AllDevList)
{
  using BufType = RawBuffer<DataType, TestType::value>;

  DataType test_data[] = {0, 0, 0, 0};
  BufType buf = BufType(test_data, 4);

  REQUIRE(buf.size() == 4);
  REQUIRE(buf.data() == test_data);
  REQUIRE(buf.const_data() == test_data);
  buf.ensure();
  REQUIRE(buf.data() == test_data);
  REQUIRE(buf.const_data() == test_data);
  buf.release();
  REQUIRE(buf.data() == nullptr);
  REQUIRE(buf.const_data() == nullptr);
  buf.ensure();
  REQUIRE(buf.data() != nullptr);
  REQUIRE(buf.const_data() != nullptr);
  REQUIRE(buf.data() != test_data);
  REQUIRE(buf.const_data() != test_data);
}

TEMPLATE_LIST_TEST_CASE("Raw buffers are writable",
                        "[tensor][raw_buffer]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using BufType = RawBuffer<DataType, TestType::value>;
  constexpr std::size_t buf_size = 32;

  BufType buf = BufType(buf_size);
  REQUIRE(buf.size() == buf_size);
  REQUIRE(buf.data() != nullptr);

  // There are no assertions here, but hopefully things crash and burn
  // if this doesn't work (or a sanitizer catches it).
  DataType* raw_buf = buf.data();
  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<Dev>(raw_buf, i, static_cast<DataType>(i));
  }

  // Ensure on already allocated data should not change anything.
  buf.ensure();
  REQUIRE(buf.size() == buf_size);
  REQUIRE(buf.data() == raw_buf);
  for (std::size_t i = 0; i < buf_size; ++i)
  {
    REQUIRE(read_ele<Dev>(raw_buf, i) == i);
  }

  // Release then ensure should be sane, but has no guarantees about
  // getting the same buffer.
  buf.release();
  REQUIRE(buf.size() == buf_size);
  REQUIRE(buf.data() == nullptr);
  buf.ensure();
  REQUIRE(buf.size() == buf_size);
  REQUIRE(buf.data() != nullptr);
  raw_buf = buf.data();
  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<Dev>(raw_buf, i, static_cast<DataType>(i));
  }
}
