////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Lazy.hpp"

namespace h2
{
namespace meta
{
/** @brief A SFINAE tool for excluding functions/overloads.
 *
 *  Contains a typedef `type` if the condition is `true`.
 */
template <bool B, typename ResultT = void>
struct EnableIfT;

/** @brief A SFINAE tool that contains a type when the condition is true. */
template <bool B, typename ResultT = void>
using EnableIf = meta::Force<EnableIfT<B, ResultT>>;

/** @brief An alias for EnableIf. */
template <bool B, typename ResultT = void>
using EnableWhen = EnableIf<B, ResultT>;

/** @brief A SFINAE tool that contains a type when the condition is false. */
template <bool B, typename ResultT = void>
using EnableUnless = EnableWhen<!B, ResultT>;

/** @brief A version of EnableIf that operates on valued types. */
template <typename B, typename ResultT = void>
using EnableIfV = EnableIf<B::value, ResultT>;

/** @brief An alias for EnableIfV. */
template <typename B, typename ResultT = void>
using EnableWhenV = EnableWhen<B::value, ResultT>;

/** @brief A version of EnableUnless that operates on valued types. */
template <typename B, typename ResultT = void>
using EnableUnlessV = EnableUnless<B::value, ResultT>;

/** @brief Representation of a substitution failure.
 *
 *  This follows an idiom I first encountered in _The C++ Programming
 *  Language_ by Bjarne Stroustrop.
 */
struct SubstitutionFailure;

/** @brief Representation of a substitution success. */
template <typename T>
struct SubstitutionSuccess
{
  static constexpr bool value = true;
};

/** @brief Substitution failure is not success. */
template <>
struct SubstitutionSuccess<SubstitutionFailure>
{
  static constexpr bool value = false;
};

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <bool B, typename ResultT>
struct EnableIfT
{};

template <typename ResultT>
struct EnableIfT<true, ResultT>
{
  using type = ResultT;
};

#endif // DOXYGEN_SHOULD_SKIP_THIS

} // namespace meta
} // namespace h2
