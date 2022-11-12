////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_SORT_HPP_
#define H2_META_TYPELIST_SORT_HPP_

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/IfThenElse.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Sort a typelist using a user-provided comparison function.
 *
 *  If `Compare<T1, T2>` is True, then T1 is ordered before T2. The
 *  ordering is important.
 *
 *  Sorting the empty list returns the empty list.
 */
template <typename List, template <typename, typename> class Compare>
struct SortT;

/** @brief Sort a typelist using a user-provided comparison function. */
template <typename List, template <typename, typename> class Compare>
using Sort = Force<SortT<List, Compare>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace details
{
template <typename T,
          typename SortedList,
          template <typename, typename>
          class Compare>
struct InsertIntoSortedT;

template <typename T,
          typename SortedList,
          template <typename, typename>
          class Compare>
using InsertIntoSorted = Force<InsertIntoSortedT<T, SortedList, Compare>>;

template <typename T, template <typename, typename> class Compare>
struct InsertIntoSortedT<T, Empty, Compare>
{
    using type = TL<T>;
};

template <typename T,
          typename Head,
          typename... Tail,
          template <typename, typename>
          class Compare>
struct InsertIntoSortedT<T, TL<Head, Tail...>, Compare>
    : IfThenElseT<Compare<T, Head>::value,
                  TL<T, Head, Tail...>,
                  Cons<Head, InsertIntoSorted<T, TL<Tail...>, Compare>>>
{};

} // namespace details

template <template <typename, typename> class Compare>
struct SortT<Empty, Compare>
{
    using type = Empty;
};

// Insertion sort -- it works and is straightforward to implement. If
// lists grow large, we can reevaluate this choice.
template <typename Head,
          typename... Tail,
          template <typename, typename>
          class Compare>
struct SortT<TL<Head, Tail...>, Compare>
    : details::InsertIntoSortedT<Head, Sort<TL<Tail...>, Compare>, Compare>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_SORT_HPP_
