// @H2_LICENSE_TEXT@

#ifndef H2_META_TYPELIST_LENGTH_HPP_
#define H2_META_TYPELIST_LENGTH_HPP_

#include "LispAccessors.hpp"
#include "TypeList.hpp"

#include "h2/meta/core/Lazy.hpp"
#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
/** @brief Get the index of a given type in the list. */
template <typename List>
struct LengthVT;

/** @brief Get the index of a given type in the list. */
template <typename List>
constexpr unsigned long LengthV() { return LengthVT<List>::value; }

#ifndef H2_USE_CXX17
template <typename List>
inline constexpr unsigned long Length = LengthV<List>();
#endif // H2_USE_CXX17

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <>
struct LengthVT<Empty>
    : ValueAsType<unsigned long, 0>
{};

// Recursive case
template <typename T, typename... Ts>
struct LengthVT<TL<T, Ts...>>
    : ValueAsType<unsigned long, 1 + LengthV<TL<Ts...>>()>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_LENGTH_HPP_
