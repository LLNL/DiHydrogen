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

#include "h2/utils/const_for.hpp"
#include "h2/utils/function_traits.hpp"

#include <cstddef>
#include <type_traits>

namespace h2
{
namespace cpu
{

/**
 * Naive n-ary element-wise loop.
 *
 * If func returns a value, the first pointer in args is required to be
 * an output buffer of a type which func's return type is convertible
 * to. (The return value may not be discarded.)
 */
template <typename FuncT, typename... Args>
void elementwise_loop(FuncT&& func, std::size_t size, Args... args)
{
  using traits = FunctionTraits<FuncT>;
  constexpr bool has_return = !std::is_same_v<typename traits::RetT, void>;
  constexpr std::size_t arg_offset = has_return ? 1 : 0;
  static_assert(traits::arity + arg_offset == sizeof...(args),
                "Argument number mismatch");
  // TODO: Check args is convertible to function args.

  std::tuple<Args...> args_ptrs{args...};
  meta::tlist::ToTuple<typename traits::ArgsList> loaded_args;

  static_assert(
    !has_return
      || std::is_convertible_v<
        typename traits::RetT,
        std::remove_pointer_t<std::tuple_element_t<0, decltype(args_ptrs)>>>,
    "Cannt convert return value to output");

  for (std::size_t i = 0; i < size; ++i)
  {
    const_for<arg_offset, sizeof...(Args), std::size_t{1}>([&](auto arg_i) {
      std::get<arg_i.value - arg_offset>(loaded_args) =
        std::get<arg_i.value>(args_ptrs)[i];
    });
    if constexpr (has_return)
    {
      std::get<0>(args_ptrs)[i] = std::apply(func, loaded_args);
    }
    else
    {
      std::apply(func, loaded_args);
    }
  }
}

}  // namespace cpu
}  // namespace h2
