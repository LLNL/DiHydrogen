////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

namespace h2
{
namespace meta
{
/** @brief Suspend a given type. */
template <typename T>
struct Susp
{
    using type = T;
};

/** @brief Extract the internal type from a suspended type. */
template <typename SuspT>
using Force = typename SuspT::type;

} // namespace meta
} // namespace h2
