////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "TypeList.hpp"
#include "h2/meta/core/IfThenElse.hpp"
#include "h2/meta/core/ValueAsType.hpp"

namespace h2
{
namespace meta
{
namespace tlist
{
constexpr static unsigned long InvalidIdx = static_cast<unsigned long>(-1);

/** @brief Get the index of a given type in the list. */
template <typename List, typename T>
struct FindVT;

/** @brief Get the index of a given type in the list. */
template <typename List, typename T>
constexpr unsigned long FindV()
{
  return FindVT<List, T>::value;
}

template <typename List, typename T>
inline constexpr unsigned long Find = FindV<List, T>();

/** @brief Get the index of the first type matching Predicate in the list. */
template <template <typename> class Pred, typename List>
struct FindIfVT;

/** @brief Get the index of the first type matching Predicate in the list. */
template <template <typename> class Pred, typename List>
constexpr unsigned long FindIfV()
{
  return FindIfVT<Pred, List>::value;
}

/** @brief Get the index of the first type matching Predicate in the list. */
template <template <typename> class Pred, typename List>
inline constexpr unsigned long FindIf = FindIfV<Pred, List>();

#ifndef DOXYGEN_SHOULD_SKIP_THIS

template <typename List, typename T, unsigned long N>
struct FindVTImpl;

template <typename T, unsigned long N>
struct FindVTImpl<Empty, T, N> : ValueAsType<unsigned long, InvalidIdx>
{};

template <typename... Ts, typename T, unsigned long N>
struct FindVTImpl<TL<T, Ts...>, T, N> : ValueAsType<unsigned long, N>
{};

template <typename... Ts, typename S, typename T, unsigned long N>
struct FindVTImpl<TL<S, Ts...>, T, N> : FindVTImpl<TL<Ts...>, T, N + 1>
{};

template <typename List, typename T>
struct FindVT : FindVTImpl<List, T, 0UL>
{};

template <template <typename> class Pred, typename List, unsigned long N>
struct FindIfVTImpl;

template <template <typename> class Pred, unsigned long N>
struct FindIfVTImpl<Pred, Empty, N> : ValueAsType<unsigned long, InvalidIdx>
{};

template <template <typename> class Pred,
          typename T,
          typename... Ts,
          unsigned long N>
struct FindIfVTImpl<Pred, TL<T, Ts...>, N>
  : IfThenElse<Pred<T>::value,
               ValueAsType<unsigned long, N>,
               FindIfVTImpl<Pred, TL<Ts...>, N + 1>>
{};

template <template <typename> class Pred, typename List>
struct FindIfVT : FindIfVTImpl<Pred, List, 0UL>
{};

#endif  // DOXYGEN_SHOULD_SKIP_THIS
}  // namespace tlist
}  // namespace meta
}  // namespace h2
