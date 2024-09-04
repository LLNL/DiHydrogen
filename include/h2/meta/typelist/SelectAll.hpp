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

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Select all types from a list that match the predicate.
 *
 *  If no type matches, the empty list is returned.
 */
template <typename List, template <typename> class Predicate>
struct SelectAllT;

/** @brief Select all types from a list that match the predicate. */
template <typename List, template <typename> class Predicate>
using SelectAll = Force<SelectAllT<List, Predicate>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <typename> class Predicate>
struct SelectAllT<Empty, Predicate>
{
    using type = Empty;
};

template <typename T, typename... Ts, template <typename> class Predicate>
struct SelectAllT<TL<T, Ts...>, Predicate>
{
private:
    static constexpr auto Value_ = Predicate<T>::value;
    using Rest_ = SelectAll<TL<Ts...>, Predicate>;

public:
    using type = IfThenElse<Value_, Cons<T, Rest_>, Rest_>;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
