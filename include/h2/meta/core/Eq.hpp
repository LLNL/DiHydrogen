////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_CORE_EQ_HPP_
#define H2_META_CORE_EQ_HPP_

#include "Lazy.hpp"
#include "ValueAsType.hpp"

namespace h2
{
namespace meta
{
/** @brief Binary metafunction for type equality. */
template <typename T, typename U>
struct EqVT;

template <typename T, typename U>
inline constexpr bool EqV()
{
    return EqVT<T, U>::value;
}

template <typename T, typename U>
inline constexpr bool Eq = EqV<T, U>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T, typename U>
struct EqVT : FalseType
{};

template <typename T>
struct EqVT<T, T> : TrueType
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace meta
} // namespace h2
#endif // H2_META_CORE_EQ_HPP_
