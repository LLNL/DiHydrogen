// @H2_LICENSE_TEXT@

#ifndef H2_META_TYPELIST_REMOVEALL_HPP_
#define H2_META_TYPELIST_REMOVEALL_HPP_

#include "LispAccessors.hpp"
#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Remove all instances of a type from a typelist. */
template <typename List, typename T>
struct RemoveAllT;

/** @brief Remove all instances of a type from a typelist. */
template <typename List, typename T>
using RemoveAll = Force<RemoveAllT<List, T>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base Case
template <typename T>
struct RemoveAllT<Empty, T>
{
    using type = Empty;
};

// Match case
template <typename T, typename... Ts>
struct RemoveAllT<TypeList<T, Ts...>, T> : RemoveAllT<TypeList<Ts...>, T>
{};

// Recursive call
template <typename S, typename... Ts, typename T>
struct RemoveAllT<TypeList<S, Ts...>, T>
    : ConsT<S, RemoveAll<TypeList<Ts...>, T>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_REMOVEALL_HPP_
