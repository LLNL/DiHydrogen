////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_LENGTH_HPP_
#define H2_META_TYPELIST_LENGTH_HPP_

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
/** @brief Get the index of a given type in the list. */
template <typename List>
struct LengthVT;

/** @brief Get the index of a given type in the list. */
template <typename List>
constexpr unsigned long LengthV()
{
    return LengthVT<List>::value;
}

template <typename List>
inline constexpr unsigned long Length = LengthV<List>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <>
struct LengthVT<Empty> : ValueAsType<unsigned long, 0>
{};

// Recursive case
template <typename T, typename... Ts>
struct LengthVT<TL<T, Ts...>>
    : ValueAsType<unsigned long, 1 + LengthV<TL<Ts...>>()>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_LENGTH_HPP_
