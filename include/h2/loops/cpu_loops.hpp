////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

// This file is meant to be included only in source files.

#pragma once

/** @file
 *
 * Loop routines for CPUs.
 */

#include <h2_config.hpp>

#include <cstddef>
#include <type_traits>

#include "h2/utils/const_for.hpp"
#include "h2/utils/function_traits.hpp"


namespace h2
{
namespace cpu
{

template <typename FuncT, typename... Args>
void inplace_elementwise_loop(FuncT&& func,
                              std::size_t size,
                              Args* __restrict__... args)
{
  using traits = FunctionTraits<FuncT>;
  static_assert(traits::arity == sizeof...(args), "Argument number mismatch");
  static_assert(std::is_same_v<typename traits::RetT, void>,
                "Would discard return");
  // TODO: Check args is convertible to function args.

  std::tuple<Args* __restrict__...> args_ptrs{args...};
  typename traits::ArgsTuple loaded_args;

  for (std::size_t i = 0; i < size; ++i)
  {
    const_for<std::size_t{0}, sizeof...(Args), std::size_t{1}>([&](auto arg_i) {
      std::get<arg_i.value>(loaded_args) = std::get<arg_i.value>(args_ptrs)[i];
    });
    std::apply(func, loaded_args);
  }
}

template <typename FuncT, typename OutT, typename... Args>
void elementwise_loop(FuncT&& func,
                      std::size_t size,
                      OutT* __restrict__ out,
                      Args* __restrict__... args)
{
  using traits = FunctionTraits<FuncT>;
  static_assert(traits::arity == sizeof...(args), "Argument number mismatch");
  static_assert(std::is_convertible_v<typename traits::RetT, OutT>,
                "Cannot convert return value to output type");
  // TODO: Check args is convertible to function args.

  std::tuple<Args* __restrict__...> args_ptrs{args...};
  typename traits::ArgsTuple loaded_args;

  for (std::size_t i = 0; i < size; ++i)
  {
    const_for<std::size_t{0}, sizeof...(Args), std::size_t{1}>([&](auto arg_i) {
      std::get<arg_i.value>(loaded_args) = std::get<arg_i.value>(args_ptrs)[i];
    });
    out[i] = std::apply(func, loaded_args);
  }
}

}  // namespace cpu
}  // namespace h2
