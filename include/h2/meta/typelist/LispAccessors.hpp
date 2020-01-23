// @H2_LICENSE_TEXT@
#ifndef H2_META_TYPELIST_LISPACCESSORS_HPP_
#define H2_META_TYPELIST_LISPACCESSORS_HPP_

#include "TypeList.hpp"
#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief The basic Cons operation.
 *  @details Prepend an item to a list.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename T, typename List>
struct ConsT;

/** @brief An appending version of the Cons operation.
 *  @details A naive lisp implementation makes this an O(n) operation;
 *           however, the nature of variadic templates allows an O(1)
 *           implementation. Thus it is provided as a convenience.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename List, typename T>
struct ConsBackT;

/** @brief Get the first item in a list. */
template <typename List>
struct CarT;

/** @brief Get a copy of the list with the first item removed. */
template <typename List>
struct CdrT;

/** @brief The basic Cons operation.
 *  @details Prepend an item to a list.
 *  @tparam T The new item to prepend to the list
 *  @tparam List The list
 */
template <typename T, typename List>
using Cons = Force<ConsT<T, List>>;

/** @brief An appending version of the Cons operation.
 *  @details Append an item to a list.
 *  @tparam List The list
 *  @tparam T The new item to prepend to the list
 */
template <typename List, typename T>
using ConsBack = Force<ConsBackT<List, T>>;

/** @brief Get the first item in a list
 *  @tparam List The list.
 */
template <typename List>
using Car = Force<CarT<List>>;

/** @brief Get a copy of the list with the first item removed
 *  @tparam List The list
 */
template <typename List>
using Cdr = Force<CdrT<List>>;

// A few Lisp-y things. The CL spec goes out to 4 operations.

// 2 operations
template <typename List>
using Caar = Car<Car<List>>;
template <typename List>
using Cadr = Car<Cdr<List>>;
template <typename List>
using Cdar = Cdr<Car<List>>;
template <typename List>
using Cddr = Cdr<Cdr<List>>;

// 3 operations
template <typename List>
using Caaar = Car<Caar<List>>;
template <typename List>
using Caadr = Car<Cadr<List>>;
template <typename List>
using Cadar = Car<Cdar<List>>;
template <typename List>
using Cdaar = Cdr<Caar<List>>;
template <typename List>
using Caddr = Car<Cddr<List>>;
template <typename List>
using Cddar = Cdr<Cdar<List>>;
template <typename List>
using Cdadr = Cdr<Cadr<List>>;
template <typename List>
using Cdddr = Cdr<Cddr<List>>;

// 4 operations
template <typename List>
using Caaaar = Car<Caaar<List>>;
template <typename List>
using Caaadr = Car<Caadr<List>>;
template <typename List>
using Caadar = Car<Cadar<List>>;
template <typename List>
using Cadaar = Car<Cdaar<List>>;
template <typename List>
using Cdaaar = Cdr<Caaar<List>>;
template <typename List>
using Caaddr = Car<Caddr<List>>;
template <typename List>
using Cadadr = Car<Cdadr<List>>;
template <typename List>
using Cdaadr = Cdr<Caadr<List>>;
template <typename List>
using Cdadar = Cdr<Cadar<List>>;
template <typename List>
using Cddaar = Cdr<Cdaar<List>>;
template <typename List>
using Caddar = Car<Cddar<List>>;
template <typename List>
using Cadddr = Car<Cdddr<List>>;
template <typename List>
using Cdaddr = Cdr<Caddr<List>>;
template <typename List>
using Cddadr = Cdr<Cdadr<List>>;
template <typename List>
using Cdddar = Cdr<Cddar<List>>;
template <typename List>
using Cddddr = Cdr<Cdddr<List>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Cons
template <typename T, typename... Ts>
struct ConsT<T, TypeList<Ts...>>
{
    using type = TypeList<T, Ts...>;
};

// ConsBack
template <typename T, typename... Ts>
struct ConsBackT<TypeList<Ts...>, T>
{
    using type = TypeList<Ts..., T>;
};

// Car
template <typename T, typename... Ts>
struct CarT<TypeList<T, Ts...>>
{
    using type = T;
};

template <>
struct CarT<Empty>
{
    using type = Nil;
};

// Cdr
template <typename T, typename... Ts>
struct CdrT<TypeList<T, Ts...>>
{
    using type = TypeList<Ts...>;
};

template <>
struct CdrT<Empty>
{
    using type = Empty;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace tlist
} // namespace meta
} // namespace h2
#endif // H2_META_TYPELIST_LISPACCESSORS_HPP_
