////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Lazy.hpp"

namespace h2
{
namespace meta
{
template <bool B, typename T, typename F>
struct IfThenElseT;

template <bool B, typename T, typename F>
using IfThenElse = Force<IfThenElseT<B, T, F>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS
template <bool B, typename T, typename F>
struct IfThenElseT
{
  using type = F;
};

template <typename T, typename F>
struct IfThenElseT<true, T, F>
{
  using type = T;
};
#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace meta
}  // namespace h2
