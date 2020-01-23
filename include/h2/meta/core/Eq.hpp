// @H2_LICENSE_TEXT@

#ifndef H2_META_CORE_EQ_HPP_
#define H2_META_CORE_EQ_HPP_

#include "Lazy.hpp"
#include "ValueAsType.hpp"

namespace h2
{
namespace meta
{
/** @brief Binary metafunction for type equality. */
template <typename T, typename U>
struct EqVT;

template <typename T, typename U>
inline constexpr bool EqV()
{
    return EqVT<T, U>::value;
}

#ifndef H2_NO_CPP17
template <typename T, typename U>
inline constexpr bool Eq = EqV<T, U>();
#endif // H2_NO_CPP17

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename T, typename U>
struct EqVT : FalseType
{};

template <typename T>
struct EqVT<T, T> : TrueType
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace meta
} // namespace h2
#endif // H2_META_CORE_EQ_HPP_
