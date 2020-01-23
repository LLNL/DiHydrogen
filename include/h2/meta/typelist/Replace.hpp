// @H2_LICENSE_TEXT@

#ifndef H2_META_TYPELIST_REPLACE_HPP_
#define H2_META_TYPELIST_REPLACE_HPP_

#include "TypeList.hpp"
#include "LispAccessors.hpp"

#include "h2/meta/core/Lazy.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Replace the first instance of Old with New.
 *  @tparam List The list in which to do replacement.
 *  @tparam Old The type to be replaced.
 *  @tparam New The replacement type.
 */
template <typename List, typename Old, typename New>
struct ReplaceT;

/** @brief Replace the first instance of Old with New. */
template <typename List, typename Old, typename New>
using Replace = Force<ReplaceT<List, Old, New>>;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename Old, typename New>
struct ReplaceT<Empty, Old, New>
{
    using type = Empty;
};

// Replacement case
template <typename Old, typename New, typename... Ts>
struct ReplaceT<TypeList<Old, Ts...>, Old, New>
{
    using type = TypeList<New, Ts...>;
};

// Recursive case
template <typename List, typename Old, typename New>
struct ReplaceT
    : ConsT<Car<List>, Replace<Cdr<List>, Old, New>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_REPLACE_HPP_
