////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "h2/utils/typename.hpp"


namespace h2
{
namespace meta
{
namespace tlist
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename L>
struct PrintT;

template <>
struct PrintT<Empty>
{
  static std::string to_string() { return ""; }
};

template <typename T>
struct PrintT<TL<T>>
{
  static std::string to_string() { return TypeName<T>(); }
};

template <typename T, typename... Ts>
struct PrintT<TL<T, Ts...>>
{
  static std::string to_string()
  {
    return TypeName<T>() + ", " + PrintT<TL<Ts...>>::to_string();
  }
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Convert a type list to a string. */
template <typename... Ts>
std::string print(const TL<Ts...>&)
{
  return PrintT<TL<Ts...>>::to_string();
}

}  // namespace tlist
}  // namespace meta
}  // namespace h2
