////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/loops/cpu_loops.hpp"
#include "../tensor/utils.hpp"


using namespace h2;


TEMPLATE_LIST_TEST_CASE("CPU elementwise loop works",
                        "[loops]",
                        h2::ComputeTypes)
{
  using Type = TestType;

  SECTION("Empty buffer")
  {
    // No assertions, if this compiles and doesn't segfault, we should
    // be okay.
    Type* empty_buf = nullptr;
    cpu::elementwise_loop([](Type) {}, 0, empty_buf);
    cpu::elementwise_loop(
        []() -> Type { return static_cast<Type>(42); }, 0, empty_buf);
    cpu::elementwise_loop([](Type) -> Type { return static_cast<Type>(42); },
                          0,
                          empty_buf,
                          empty_buf);
  }

  SECTION("Zero buffers and return")
  {
    DeviceBuf<Type, Device::CPU> buf{32};
    buf.fill(static_cast<Type>(0));
    const Type val = static_cast<Type>(42);
    cpu::elementwise_loop([&]() { return val; }, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(buf.buf[i] == static_cast<Type>(42));
    }
  }

  SECTION("One buffer")
  {
    DeviceBuf<Type, Device::CPU> buf{32};
    buf.fill(static_cast<Type>(42));
    // Has no effect.
    cpu::elementwise_loop([](Type v) {}, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(buf.buf[i] == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return")
  {
    DeviceBuf<Type, Device::CPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    cpu::elementwise_loop([](Type v) -> Type { return v + 1; },
                          out_buf.size,
                          out_buf.buf,
                          static_cast<const Type*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(in_buf.buf[i] == static_cast<Type>(42));
      REQUIRE(out_buf.buf[i] == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, no return")
  {
    DeviceBuf<Type, Device::CPU> buf1{32}, buf2{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    // Has no effect.
    cpu::elementwise_loop([](Type v1, Type v2) { v1 += v2; },
                          buf1.size,
                          buf1.buf,
                          static_cast<const Type*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(buf1.buf[i] == static_cast<Type>(21));
      REQUIRE(buf2.buf[i] == static_cast<Type>(21));
    }
  }

  SECTION("Two buffers, and return")
  {
    DeviceBuf<Type, Device::CPU> buf1{32}, buf2{32}, out_buf{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    cpu::elementwise_loop([](Type v1, Type v2) -> Type { return v1 + v2; },
                          buf1.size,
                          out_buf.buf,
                          static_cast<const Type*>(buf1.buf),
                          static_cast<const Type*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(buf1.buf[i] == static_cast<Type>(21));
      REQUIRE(buf2.buf[i] == static_cast<Type>(21));
      REQUIRE(out_buf.buf[i] == static_cast<Type>(42));
    }
  }
}
