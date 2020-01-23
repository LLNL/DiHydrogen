// @H2_LICENSE_TEXT@

#ifndef H2_META_TYPELIST_MEMBER_HPP_
#define H2_META_TYPELIST_MEMBER_HPP_

#include "TypeList.hpp"

#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{

/** @brief Determine if T is a member of List. */
template <typename T, typename List>
struct MemberVT;

/** @brief Determine if T is a member of List. */
template <typename T, typename List>
constexpr bool MemberV() { return MemberVT<T, List>::value; }

#ifdef H2_USE_CXX17
template <typename T, typename List>
inline constexpr bool Member = MemberV<T,List>();
#endif // H2_USE_CXX17

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// Base case
template <typename T>
struct MemberVT<T, Empty>
  : FalseType
{};

// Match case
template <typename T, typename... Ts>
struct MemberVT<T, TL<T, Ts...>>
  : TrueType
{};

// Recursive case
template <typename T, typename Head, typename... Tail>
struct MemberVT<T, TL<Head, Tail...>>
  : MemberVT<T, TL<Tail...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
}// namespace tlist
}// namespace meta
}// namespace h2
#endif // H2_META_TYPELIST_MEMBER_HPP_
