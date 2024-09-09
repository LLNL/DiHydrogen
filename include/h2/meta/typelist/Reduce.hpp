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

#include <type_traits>

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Apply a left fold on a type list. */
template <template <class, class> class F, typename Acc, typename List>
struct FoldlTLT;

/** @brief Apply a left fold on a type list. */
template <template <class, class> class F, typename Acc, typename List>
using FoldlTL = Force<FoldlTLT<F, Acc, List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <class, class> class F, typename Acc>
struct FoldlTLT<F, Acc, Empty>
{
  using type = Acc;
};

template <template <class, class> class F, typename Acc, typename... Ts>
struct FoldlTLT<F, Acc, TL<Ts...>>
{
  using type = FoldlTL<F, Force<F<Acc, Car<TL<Ts...>>>>, Cdr<TL<Ts...>>>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

/** @brief Apply a right fold on a type list. */
template <template <class, class> class F, typename Acc, typename... Ts>
struct FoldrTLT;

/** @brief Apply a right fold on a type list. */
template <template <class, class> class F, typename Acc, typename List>
using FoldrTL = Force<FoldrTLT<F, Acc, List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <template <class, class> class F, typename Acc>
struct FoldrTLT<F, Acc, Empty>
{
  using type = Acc;
};

template <template <class, class> class F, typename Acc, typename... Ts>
struct FoldrTLT<F, Acc, TL<Ts...>>
{
  using type = FoldrTL<F, Force<F<Car<TL<Ts...>>, Acc>>, Cdr<TL<Ts...>>>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace tlist

/** @brief Logical And between two types. */
template <typename T, typename U>
struct And;

/** @brief Logical Or between two types. */
template <typename T, typename U>
struct Or;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <bool A, bool B>
struct And<std::bool_constant<A>, std::bool_constant<B>>
{
  using type = std::bool_constant<A && B>;
};

template <bool A, bool B>
struct Or<std::bool_constant<A>, std::bool_constant<B>>
{
  using type = std::bool_constant<A || B>;
};

#endif  // DOXYGEN_SHOULD_SKIP_THIS

}  // namespace meta
}  // namespace h2
