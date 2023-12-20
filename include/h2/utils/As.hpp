////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/meta/Core.hpp"

#include <stdexcept>
#include <type_traits>

namespace h2
{

// Short-hand for static_cast
template <typename To,
          typename From,
          meta::EnableWhen<std::is_scalar_v<From>, int> = 1>
constexpr To as(From const& x)
{
    return static_cast<To>(x);
}

/** @brief Checked potentially narrowing cast.
 *
 *  @tparam To The desired output type.
 *  @tparam From (Inferred) The input type.
 *
 *  @param[in] x The value to convert.
 *
 *  @throws std::runtime_error Thrown when the input value is not
 *                             representable in the output type.
 */
template <typename To,
          typename From,
          meta::EnableWhen<std::is_scalar_v<From>, int> = 1>
constexpr To safe_as(From const& x)
{
    constexpr bool signs_differ =
        (std::is_signed_v<To> != std::is_signed_v<From>);
    auto tmp = as<To>(x);
    if (as<From>(tmp) != x
        || (signs_differ && (tmp < To{0}) != (x < From{0})))
    {
        throw std::runtime_error("narrowing cast failed.");
    }
    return tmp;
}

} // namespace h2
