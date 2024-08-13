////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#ifndef H2_META_TYPELIST_REDUCE_HPP_
#define H2_META_TYPELIST_REDUCE_HPP_

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

template <template <class,class> class F, typename Acc, typename List>
struct FoldlTLT;

template <template <class,class> class F, typename Acc, typename List>
using FoldlTL = Force<FoldlTLT<F, Acc, List>>;

template <template <class,class> class F, typename Acc>
struct FoldlTLT<F, Acc, Empty>
{
    using type = Acc;
};

template <template <class, class> class F, typename Acc, typename... Ts>
struct FoldlTLT<F, Acc, TL<Ts...>>
{
    using type = FoldlTL<F, Force<F<Acc, Car<TL<Ts...>>>>, Cdr<TL<Ts...>>>;
};

template <template <class,class> class F, typename Acc, typename... Ts>
struct FoldrTLT;

template <template <class,class> class F, typename Acc, typename List>
using FoldrTL = Force<FoldrTLT<F, Acc, List>>;

template <template <class,class> class F, typename Acc>
struct FoldrTLT<F, Acc, Empty>
{
    using type = Acc;
};

template <template <class, class> class F, typename Acc, typename... Ts>
struct FoldrTLT<F, Acc, TL<Ts...>>
{
    using type = FoldrTL<F, Force<F<Car<TL<Ts...>>, Acc>>, Cdr<TL<Ts...>>>;
};

} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_REDUCE_HPP_
