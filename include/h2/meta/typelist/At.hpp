////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_AT_HPP_
#define H2_META_TYPELIST_AT_HPP_

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Extract the type at the given index (0-based) in the list. */
template <typename List, unsigned long Idx>
struct AtT;

/**  @brief Extract the type at the given index (0-based) in the list. */
template <typename List, unsigned long Idx>
using At = Force<AtT<List, Idx>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename List>
struct AtT<List, 0UL> : CarT<List>
{};

// Recursive case
template <typename List, unsigned long Idx>
struct AtT : AtT<Cdr<List>, Idx - 1>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_AT_HPP_
