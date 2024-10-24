////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Length.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Alias for LengthVT. */
template <typename List>
using SizeVT = LengthVT<List>;

/** @brief Get the length of the given typelist. */
template <typename List>
constexpr unsigned long SizeV()
{
  return SizeVT<List>::value;
}

template <typename List>
inline constexpr unsigned long Size = SizeV<List>();

}  // namespace tlist
}  // namespace meta
}  // namespace h2
