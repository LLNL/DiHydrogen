////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

// This file is meant to be included only in source files.

#if !defined(__CUDACC__) && !defined(__HIPCC__)
#error "This file is to only be included in GPU code"
#endif

#pragma once

/** @file
 *
 * Loop routines for GPUs.
 */

#include <h2_config.hpp>

#include <type_traits>

#include <cstddef>

#include "h2/core/sync.hpp"
#include "h2/gpu/runtime.hpp"
#include "h2/gpu/macros.hpp"
//#include "h2/loops/gpu_vec_helpers.cuh"
#include "h2/utils/const_for.hpp"
#include "h2/utils/function_traits.hpp"


namespace h2
{
namespace gpu
{

namespace kernels
{

/**
 * Naive n-ary element-wise loop.
 *
 * If f returns a value, the first pointer in args is required to be
 * an output buffer of a type which f's return type is convertible to.
 * (The return value may not be discarded.)
 *
 * f is called once for each element in the input buffers, with all
 * args (except the output, if present) provided.
 */
template <typename FuncT, typename... Args>
H2_GPU_GLOBAL void elementwise_loop(const FuncT& f,
                                    std::size_t size,
                                    Args... args)
{
  using traits = FunctionTraits<FuncT>;
  constexpr bool has_return = !std::is_same_v<typename traits::RetT, void>;
  constexpr std::size_t arg_offset = has_return ? 1 : 0;
  static_assert(traits::arity + arg_offset == sizeof...(args),
                "Argument number mismatch");
  // TODO: Check args is convertible to function args.

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  std::tuple<Args...> args_ptrs{args...};
  typename traits::ArgsTuple loaded_args;

  static_assert(!has_return
                    || std::is_convertible_v<
                        typename traits::RetT,
                        std::remove_pointer_t<
                            std::tuple_element_t<0, decltype(args_ptrs)>>>,
                "Cannot convert return value to output");

  for (std::size_t i = tid; i < size; i += stride)
  {
    // Load arguments.
    const_for<arg_offset, sizeof...(Args), std::size_t{1}>([&](auto arg_i) {
      std::get<arg_i.value - arg_offset>(loaded_args) =
          std::get<arg_i.value>(args_ptrs)[i];
        });
    if constexpr (has_return)
    {
      std::get<0>(args_ptrs)[i] = std::apply(f, loaded_args);
    }
    else
    {
      std::apply(f, loaded_args);
    }
  }
}

/**
 * Like `elementwise_loop`, but taking an immediate parameter as an
 * argument.
 *
 * This allows the parameter to be passed as a kernel argument, and so
 * does not need to be in GPU memory. (Obviously, this should be kept
 * to small arguments, e.g., scalars.)
 */
template <typename FuncT, typename ImmediateT, typename... Args>
H2_GPU_GLOBAL void elementwise_loop_with_immediate(FuncT f,
                                                   std::size_t size,
                                                   ImmediateT imm,
                                                   Args... args)
{
  using traits = FunctionTraits<FuncT>;
  constexpr bool has_return = !std::is_same_v<typename traits::RetT, void>;
  constexpr std::size_t arg_offset = has_return ? 1 : 0;
  static_assert(traits::arity + arg_offset == sizeof...(args) + 1,
                "Argument number mismatch");
  static_assert(
      std::is_convertible_v<ImmediateT, typename traits::template arg<0>>,
      "Cannot pass immediate to first argument");
  // TODO: Check args is convertible to function args.

  const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = blockDim.x * gridDim.x;

  std::tuple<Args...> args_ptrs{args...};
  typename traits::ArgsTuple loaded_args;
  std::get<0>(loaded_args) = imm;  // Store immediate in first arg.

  static_assert(!has_return
                    || std::is_convertible_v<
                        typename traits::RetT,
                        std::remove_pointer_t<
                            std::tuple_element_t<0, std::tuple<Args...>>>>,
                "Cannot convert return value to output");

  for (std::size_t i = tid; i < size; i += stride)
  {
    // Load arguments.
    // This skips the output argument in args if we have a return value
    // and the immediate argument in the function's argument list.
    const_for<arg_offset, sizeof...(Args), std::size_t{1}>([&](auto arg_i) {
      std::get<arg_i.value + 1 - arg_offset>(loaded_args) =
          std::get<arg_i.value>(args_ptrs)[i];
    });
    if constexpr (has_return)
    {
      std::get<0>(args_ptrs)[i] = std::apply(f, loaded_args);
    }
    else
    {
      std::apply(f, loaded_args);
    }
  }
}

}  // namespace kernels

template <typename FuncT, typename... Args>
void launch_elementwise_loop(const FuncT& func,
                             const ComputeStream& stream,
                             std::size_t size,
                             Args... args)
{
  const unsigned int block_size = gpu::num_threads_per_block;
  const unsigned int num_blocks = (size + block_size - 1) / block_size;

  gpu::launch_kernel(kernels::elementwise_loop<FuncT, Args...>,
                     num_blocks,
                     block_size,
                     0,
                     stream.template get_stream<Device::GPU>(),
                     func,
                     size,
                     args...);
}

template <typename FuncT, typename ImmediateT, typename... Args>
void launch_elementwise_loop_with_immediate(const FuncT& func,
                                            const ComputeStream& stream,
                                            std::size_t size,
                                            ImmediateT imm,
                                            Args... args)
{
  const unsigned int block_size = gpu::num_threads_per_block;
  const unsigned int num_blocks = (size + block_size - 1) / block_size;

  gpu::launch_kernel(
      kernels::elementwise_loop_with_immediate<FuncT,
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

}  // namespace gpu
}  // namespace h2
