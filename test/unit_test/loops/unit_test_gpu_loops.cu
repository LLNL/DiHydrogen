////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/loops/gpu_loops.cuh"

#include "../tensor/utils.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

// Test helpers, use `gpu::launch_elementwise_loop`, etc. in real code.

template <typename FuncT, typename... Args>
void test_launch_naive_elementwise_loop(FuncT const func,
                                        ComputeStream const& stream,
                                        std::size_t size,
                                        Args... args)
{
  unsigned int const block_size = gpu::num_threads_per_block;
  unsigned int const num_blocks = (size + block_size - 1) / block_size;

  if (size == 0)
  {
    return;
  }

  gpu::launch_kernel(gpu::kernels::elementwise_loop<FuncT, Args...>,
                     num_blocks,
                     block_size,
                     0,
                     stream.template get_stream<Device::GPU>(),
                     func,
                     size,
                     args...);
}

template <typename FuncT, typename ImmediateT, typename... Args>
void test_launch_naive_elementwise_loop_with_immediate(
  FuncT const func,
  ComputeStream const& stream,
  std::size_t size,
  ImmediateT imm,
  Args... args)
{
  unsigned int const block_size = gpu::num_threads_per_block;
  unsigned int const num_blocks = (size + block_size - 1) / block_size;

  if (size == 0)
  {
    return;
  }

  gpu::launch_kernel(
    gpu::kernels::elementwise_loop_with_immediate<FuncT, ImmediateT, Args...>,
    num_blocks,
    block_size,
    0,
    stream.template get_stream<Device::GPU>(),
    func,
    size,
    imm,
    args...);
}

template <std::size_t vec_width,
          std::size_t unroll_factor,
          typename FuncT,
          typename... Args>
void test_launch_vectorized_elementwise_loop(FuncT const func,
                                             ComputeStream const& stream,
                                             std::size_t size,
                                             Args... args)
{
  unsigned int const block_size = gpu::num_threads_per_block;
  unsigned int const num_blocks = (size + block_size - 1) / block_size;

  if (size == 0)
  {
    return;
  }

  std::size_t reqd_vec_width =
    std::min({gpu::max_vectorization_amount(args)...});
  if (vec_width > reqd_vec_width)
  {
    throw H2FatalException(
      "Invalid test vector width: ", vec_width, " > ", reqd_vec_width);
  }

  gpu::launch_kernel(gpu::kernels::vectorized_elementwise_loop<std::size_t,
                                                               vec_width,
                                                               unroll_factor,
                                                               FuncT,
                                                               Args...>,
                     num_blocks,
                     block_size,
                     0,
                     stream.template get_stream<Device::GPU>(),
                     func,
                     size,
                     args...);
}

template <std::size_t vec_width,
          std::size_t unroll_factor,
          typename FuncT,
          typename ImmediateT,
          typename... Args>
void test_launch_vectorized_elementwise_loop_with_immediate(
  FuncT const func,
  ComputeStream const& stream,
  std::size_t size,
  ImmediateT imm,
  Args... args)
{
  unsigned int const block_size = gpu::num_threads_per_block;
  unsigned int const num_blocks = (size + block_size - 1) / block_size;

  if (size == 0)
  {
    return;
  }

  std::size_t reqd_vec_width =
    std::min({gpu::max_vectorization_amount(args)...});
  if (vec_width > reqd_vec_width)
  {
    throw H2FatalException(
      "Invalid test vector width: ", vec_width, " > ", reqd_vec_width);
  }

  gpu::launch_kernel(
    gpu::kernels::vectorized_elementwise_loop_with_immediate<std::size_t,
                                                             vec_width,
                                                             unroll_factor,
                                                             FuncT,
                                                             ImmediateT,
                                                             Args...>,
    num_blocks,
    block_size,
    0,
    stream.template get_stream<Device::GPU>(),
    func,
    size,
    imm,
    args...);
}

TEMPLATE_LIST_TEST_CASE("Naive GPU element-wise loop works",
                        "[loops]",
                        h2::ComputeTypes)
{
  using Type = TestType;

  ComputeStream stream{Device::GPU};

  SECTION("Empty buffer")
  {
    // No assertions, if this compiles and doesn't segfault, we should
    // be okay.
    Type* empty_buf = nullptr;
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type) {}, stream, 0, empty_buf);
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA() -> Type { return static_cast<Type>(42); },
      stream,
      0,
      empty_buf);
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type) -> Type { return static_cast<Type>(42); },
      stream,
      0,
      empty_buf,
      empty_buf);
  }

  SECTION("Zero buffers and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA() { return 42; }, stream, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("Zero buffers, capture-by-value, and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    Type const val = static_cast<Type>(42);
    test_launch_naive_elementwise_loop(
      [val] H2_GPU_LAMBDA() { return val; }, stream, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(42));
    // Has no effect.
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type v) {}, stream, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, no return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    // Has no effect.
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type v1, Type v2) { v1 += v2; },
      stream,
      buf1.size,
      buf1.buf,
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
    }
  }

  SECTION("Two buffers, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32}, out_buf{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      buf1.size,
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("Two buffers, funky size, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{31}, buf2{31}, out_buf{31};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      buf1.size,
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Naive GPU element-wise loop with immediate works",
                        "[loops]",
                        h2::ComputeTypes)
{
  using Type = TestType;

  ComputeStream stream{Device::GPU};

  SECTION("Zero buffers and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(Type val) { return val; },
      stream,
      buf.size,
      static_cast<Type>(42),
      buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      out_buf.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, no return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    // Has no effect.
    test_launch_naive_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(bool b, Type v1, Type v2) { v1 += b ? 42 : v2; },
      stream,
      buf1.size,
      false,
      buf1.buf,
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
    }
  }

  SECTION("Two buffers, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32}, out_buf{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(Type a, Type v1, Type v2) -> Type {
        return a + v1 + v2;
      },
      stream,
      buf1.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, funky size, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{31}, buf2{31}, out_buf{31};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_naive_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(Type a, Type v1, Type v2) -> Type {
        return a + v1 + v2;
      },
      stream,
      buf1.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Vectorized GPU element-wise loop works",
                        "[loops]",
                        h2::ComputeTypes)
{
  using Type = TestType;

  ComputeStream stream{Device::GPU};

  SECTION("Empty buffer")
  {
    // No assertions, if this compiles and doesn't segfault, we should
    // be okay.
    Type* empty_buf = nullptr;
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type) {}, stream, 0, empty_buf);
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA() -> Type { return static_cast<Type>(42); },
      stream,
      0,
      empty_buf);
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type) -> Type { return static_cast<Type>(42); },
      stream,
      0,
      empty_buf,
      empty_buf);
  }

  SECTION("Zero buffers and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA() -> Type { return static_cast<Type>(42); },
      stream,
      buf.size,
      buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("Zero buffers, capture-by-value, and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    Type const val = static_cast<Type>(42);
    test_launch_vectorized_elementwise_loop<4, 4>(
      [val] H2_GPU_LAMBDA() { return val; }, stream, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(42));
    // Has no effect.
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v) {}, stream, buf.size, buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, no return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    // Has no effect.
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v1, Type v2) { v1 += v2; },
      stream,
      buf1.size,
      buf1.buf,
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
    }
  }

  SECTION("Two buffers, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32}, out_buf{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      buf1.size,
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("Two buffers, funky size, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{31}, buf2{31}, out_buf{31};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      buf1.size,
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return, 2 unrolls")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 2>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("One buffer and return, veclen 2")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<2, 4>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("One buffer and return, tiny buffer")
  {
    DeviceBuf<Type, Device::GPU> in_buf{1}, out_buf{1};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("One buffer and return, smaller than unroll*veclen")
  {
    DeviceBuf<Type, Device::GPU> in_buf{6}, out_buf{6};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<2, 4>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("One buffer and return, big buffer")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32768}, out_buf{32768};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop<4, 4>(
      [] H2_GPU_LAMBDA(Type v) -> Type { return v + 1; },
      stream,
      out_buf.size,
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Vectorized GPU element-wise loop with immediate works",
                        "[loops]",
                        h2::ComputeTypes)
{
  using Type = TestType;

  ComputeStream stream{Device::GPU};

  SECTION("Zero buffers and return")
  {
    DeviceBuf<Type, Device::GPU> buf{32};
    buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop_with_immediate<4, 4>(
      [] H2_GPU_LAMBDA(Type val) -> Type { return val; },
      stream,
      buf.size,
      static_cast<Type>(42),
      buf.buf);
    for (std::size_t i = 0; i < buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf.buf, i, stream)
              == static_cast<Type>(42));
    }
  }

  SECTION("One buffer and return")
  {
    DeviceBuf<Type, Device::GPU> in_buf{32}, out_buf{32};
    in_buf.fill(static_cast<Type>(42));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop_with_immediate<4, 4>(
      [] H2_GPU_LAMBDA(Type v1, Type v2) -> Type { return v1 + v2; },
      stream,
      out_buf.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(in_buf.buf));
    for (std::size_t i = 0; i < in_buf.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(in_buf.buf, i, stream)
              == static_cast<Type>(42));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, no return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    // Has no effect.
    test_launch_vectorized_elementwise_loop_with_immediate<4, 4>(
      [] H2_GPU_LAMBDA(bool b, Type v1, Type v2) { v1 += b ? 42 : v2; },
      stream,
      buf1.size,
      false,
      buf1.buf,
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
    }
  }

  SECTION("Two buffers, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{32}, buf2{32}, out_buf{32};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop_with_immediate<4, 4>(
      [] H2_GPU_LAMBDA(Type a, Type v1, Type v2) -> Type {
        return a + v1 + v2;
      },
      stream,
      buf1.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }

  SECTION("Two buffers, funky size, and return")
  {
    DeviceBuf<Type, Device::GPU> buf1{31}, buf2{31}, out_buf{31};
    buf1.fill(static_cast<Type>(21));
    buf2.fill(static_cast<Type>(21));
    out_buf.fill(static_cast<Type>(0));
    test_launch_vectorized_elementwise_loop_with_immediate<4, 4>(
      [] H2_GPU_LAMBDA(Type a, Type v1, Type v2) -> Type {
        return a + v1 + v2;
      },
      stream,
      buf1.size,
      static_cast<Type>(1),
      out_buf.buf,
      static_cast<Type const*>(buf1.buf),
      static_cast<Type const*>(buf2.buf));
    for (std::size_t i = 0; i < buf1.size; ++i)
    {
      REQUIRE(read_ele<Device::GPU>(buf1.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(buf2.buf, i, stream)
              == static_cast<Type>(21));
      REQUIRE(read_ele<Device::GPU>(out_buf.buf, i, stream)
              == static_cast<Type>(43));
    }
  }
}
