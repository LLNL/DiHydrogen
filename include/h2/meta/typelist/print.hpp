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
struct PrintTLT;

template <>
struct PrintTLT<Empty>
{
  static std::string to_string() { return ""; }
};

template <typename T>
struct PrintTLT<TL<T>>
{
  static std::string to_string() { return TypeName<T>(); }
};

template <typename T, typename... Ts>
struct PrintTLT<TL<T, Ts...>>
{
  static std::string to_string()
  {
    return TypeName<T>() + ", " + PrintTLT<TL<Ts...>>::to_string();
  }
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Convert a type list to a string. */
template <typename... Ts>
std::string print(const TL<Ts...>&)
{
  return PrintTLT<TL<Ts...>>::to_string();
}

} // namespace tlist
} // namespace meta
} // namespace h2
