////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_HASKELLACCESSORS_HPP_
#define H2_META_TYPELIST_HASKELLACCESSORS_HPP_

#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Prepend an item to the list. */
template <typename T, typename List>
struct PrependT;

/** @brief Get the first item in the list. */
template <typename List>
struct HeadT;

/** @brief Extract the elements after the head of the list. */
template <typename List>
struct TailT;

/** @brief Get the last item in the list. */
template <typename List>
struct LastT;

/** @brief Extract the elements after the last element of the list. */
template <typename List>
struct InitT;

// Using aliases

/** @brief Prepend an item to the list. */
template <typename T, typename List>
using Prepend = Force<PrependT<T, List>>;

/** @brief Get the first item in the list. */
template <typename List>
using Head = Force<HeadT<List>>;

/** @brief Extract the elements after the head of the list. */
template <typename List>
using Tail = Force<TailT<List>>;

/** @brief Get the last item in the list. */
template <typename List>
using Last = Force<LastT<List>>;

/** @brief Extract the elements after the last element of the list. */
template <typename List>
using Init = Force<InitT<List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Prepend
template <typename T, typename... Ts>
struct PrependT<T, TL<Ts...>>
{
    using type = TL<T, Ts...>;
};

// Head
template <typename T, typename... Ts>
struct HeadT<TL<T, Ts...>>
{
    using type = T;
};

template <>
struct HeadT<Empty>
{
    using type = Nil;
};

// Tail
template <typename T, typename... Ts>
struct TailT<TL<T, Ts...>>
{
    using type = TL<Ts...>;
};

template <>
struct TailT<Empty>
{
    using type = Empty;
};

// Last
template <typename T>
struct LastT<TL<T>>
{
    using type = T;
};

template <typename T, typename... Ts>
struct LastT<TL<T, Ts...>> : LastT<TL<Ts...>>
{};

// Maybe this shouldn't be here, just error out?
template <>
struct LastT<Empty>
{
    using type = Nil;
};

// Init
template <typename T>
struct InitT<TL<T>>
{
    using type = Empty;
};

template <typename T, typename... Ts>
struct InitT<TL<T, Ts...>> : PrependT<T, Init<TL<Ts...>>>
{};

// Maybe this shouldn't be here, just error out?
template <>
struct InitT<Empty>
{
    using type = Empty;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_HASKELLACCESSORS_HPP_
