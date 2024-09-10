////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"
#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Get the length of the given typelist. */
template <typename List>
struct LengthVT;

/** @brief Get the length of the given typelist. */
template <typename List>
constexpr unsigned long LengthV()
{
  return LengthVT<List>::value;
}

template <typename List>
inline constexpr unsigned long Length = LengthV<List>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename... Ts>
struct LengthVT<TL<Ts...>> : ValueAsType<unsigned long, sizeof...(Ts)>
{};

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}  // namespace tlist
}  // namespace meta
}  // namespace h2
