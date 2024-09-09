////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "h2/meta/core/IfThenElse.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Select the first type from a list that matches the predicate.
 *
 *  If no type matches, Nil is returned.
 */
template <typename List, template <typename> class Predicate>
struct SelectT;

/** @brief Select the first type from a list that matches the predicate. */
template <typename List, template <typename> class Predicate>
using Select = Force<SelectT<List, Predicate>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <typename> class Predicate>
struct SelectT<Empty, Predicate>
{
  using type = Nil;
};

template <typename T, typename... Ts, template <typename> class Predicate>
struct SelectT<TL<T, Ts...>, Predicate>
  : IfThenElseT<Predicate<T>::value, T, Select<TL<Ts...>, Predicate>>
{};

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}  // namespace tlist
}  // namespace meta
}  // namespace h2
