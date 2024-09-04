////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_FIND_HPP_
#define H2_META_TYPELIST_FIND_HPP_

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
template <typename List, typename T>
struct FindVT;

/** @brief Get the index of a given type in the list. */
template <typename List, typename T>
constexpr unsigned long FindV()
{
    return FindVT<List, T>::value;
}

template <typename List, typename T>
inline constexpr unsigned long Find = FindV<List, T>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename T>
struct FindVT<Empty, T> : ValueAsType<unsigned long, -1UL>
{};

// Match case
template <typename... Ts, typename T>
struct FindVT<TL<T, Ts...>, T> : ValueAsType<unsigned long, 0>
{};

// Recursive case
template <typename... ListTs, typename T, typename U>
struct FindVT<TL<T, ListTs...>, U>
{
private:
    static constexpr auto tmp_ = FindV<TL<ListTs...>, U>();
    static constexpr auto invalid_ = static_cast<unsigned long>(-1);

public:
    static constexpr auto value = (tmp_ == invalid_ ? invalid_ : 1UL + tmp_);
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_FIND_HPP_
