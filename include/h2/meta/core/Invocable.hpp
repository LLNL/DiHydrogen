// @H2_LICENSE_TEXT@

#ifndef H2_META_CORE_INVOCABLE_HPP_
#define H2_META_CORE_INVOCABLE_HPP_

#include "SFINAE.hpp"

#include <utility>

namespace h2
{
namespace meta
{
/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
struct IsInvocableVT;

/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
inline constexpr bool IsInvocableV()
{
    return IsInvocableVT<F, Args...>::value;
}

#ifndef H2_NO_CPP17
/** @brief Test whether F can be invoked with the given arguments. */
template <typename F, typename... Args>
inline constexpr bool IsInvocable = IsInvocableV<F, Args...>();
#endif // H2_NO_CPP17

#ifndef DOXYGEN_SHOULD_SKIP_THIS

namespace details
{
// This is a detail nobody needs to see.
template <typename F, typename... Args>
struct GetInvocationResultT
{
private:
    template <typename F_deduce, typename... Args_deduce>
    static auto check(F_deduce f, Args_deduce&&... args)
        -> decltype(f(std::forward<Args_deduce>(args)...));
    static SubstitutionFailure check(...);

public:
    using type = decltype(check(std::declval<F>(), std::declval<Args>()...));
};

template <typename F, typename... Args>
using GetInvocationResult = meta::Force<GetInvocationResultT<F, Args...>>;

} // namespace details

template <typename F, typename... Args>
struct IsInvocableVT
    : SubstitutionSuccess<details::GetInvocationResult<F, Args...>>
{};

#endif // DOXYGEN_SHOULD_SKIP_THIS
} // namespace meta
} // namespace h2
#endif // H2_META_CORE_INVOCABLE_HPP_
