////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * For loops with compile-time constant indices and such.
 */

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace h2
{

/**
 * An integral for loop which provides a compile-time constant to func.
 *
 * Example:
 *     const_for<0, N, 1>([](auto i) { i is a compile-time constant })
 */
template <auto Start, auto End, auto Incr, typename FuncT>
constexpr void const_for(FuncT&& func)
{
  if constexpr (Start < End)
  {
    func(std::integral_constant<decltype(Start), Start>{});
    const_for<Start + Incr, End, Incr, FuncT>(std::forward<FuncT>(func));
  }
}

/**
 * Helper to apply a function to each element of a tuple.
 */
template <typename FuncT, typename Tuple>
constexpr void const_for_tuple(FuncT&& func, Tuple&& tuple)
{
  constexpr std::size_t size = std::tuple_size_v<std::decay_t<Tuple>>;
  const_for<std::size_t{0}, size, std::size_t{1}>(
    [&](auto i) { func(std::get<i.value>(tuple)); });
}

/**
 * Apply a function to each argument of a parameter pack.
 *
 * Example:
 *    const_for_pack([](auto const& v) { ... }, args...)
 */
template <typename FuncT, typename... Args>
constexpr void const_for_pack(FuncT&& func, Args&&... args)
{
  (func(std::forward<Args>(args)), ...);
}

}  // namespace h2
