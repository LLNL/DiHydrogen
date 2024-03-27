////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/tensor/fixed_size_tuple.hpp"

/** @file
 *
 * Utilities for fixed-size tuples.
 */

namespace h2
{

/**
 * Product reduction for a FixedSizeTuple.
 *
 * If the tuple is empty, returns start (default 1).
 */
template <typename AccT, typename T, typename SizeType, SizeType N>
constexpr AccT product(const FixedSizeTuple<T, SizeType, N>& tuple, const AccT start = AccT{1}) H2_NOEXCEPT {
  AccT r = start;
  for (SizeType i = 0; i < tuple.size(); ++i) {
    r *= static_cast<AccT>(tuple[i]);
  }
  return r;
}

/** Inner product for two FixedSizeTuples. */
template <typename AccT,
  typename T1, typename SizeType1, SizeType1 N1,
  typename T2, typename SizeType2, SizeType2 N2>
constexpr AccT inner_product(const FixedSizeTuple<T1, SizeType1, N1>& a,
                             const FixedSizeTuple<T2, SizeType2, N2>& b) H2_NOEXCEPT {
  static_assert(std::is_convertible<SizeType2, SizeType1>::value,
                "Incompatible SizeTypes");
  H2_ASSERT_DEBUG(a.size() == b.size(), "Mismatched tuple sizes");
  AccT r = AccT{0};
  for (SizeType1 i = 0; i < a.size(); ++i) {
    r += static_cast<AccT>(a[i]) * static_cast<AccT>(b[i]);
  }
  return r;
}

/** Compute an exclusive prefix product of a FixedSizeTuple. */
template <typename AccT, typename T, typename SizeType, SizeType N>
constexpr FixedSizeTuple<AccT, SizeType, N> prefix_product(
  const FixedSizeTuple<T, SizeType, N>& tuple,
  const AccT start = AccT{1}) H2_NOEXCEPT
{
  using result_t = FixedSizeTuple<AccT, SizeType, N>;
  result_t result(TuplePad<result_t>(tuple.size(), start));
  for (SizeType i = 1; i < tuple.size(); ++i)
  {
    result[i] = tuple[i-1] * result[i-1];
  }
  return result;
}

/**
 * Return true if the predicate returns true for any entry in a
 * FixedSizeTuple; otherwise return false.
 */
template <typename T, typename SizeType, SizeType N, typename Predicate>
constexpr bool any_of(const FixedSizeTuple<T, SizeType, N>& tuple,
                      Predicate p) {
  for (const auto& v : tuple)
  {
    if (p(v))
    {
      return true;
    }
  }
  return false;
}

/**
 * Return true if the predicate returns true for every entry in a
 * FixedSizeTuple; otherwise return false.
 *
 * Returns true if the tuple is empty.
 */
template <typename T, typename SizeType, SizeType N, typename Predicate>
constexpr bool all_of(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  for (const auto& v : tuple)
  {
    if (!p(v))
    {
      return false;
    }
  }
  return true;
}

/**
 * Return a new tuple that is the result of applying a predicate to
 * each entry of the tuple.
 */
template <typename T,
          typename SizeType,
          SizeType N,
          typename Predicate>
constexpr FixedSizeTuple<T, SizeType, N>
map(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<T, SizeType, N> mapped_tuple(
      TuplePad<FixedSizeTuple<T, SizeType, N>>(tuple.size()));
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    mapped_tuple[i] = p(tuple[i]);
  }
  return mapped_tuple;
}

/** Map with a different FixedSizeTuple return type. */
template <typename NewT,
          typename T,
          typename SizeType,
          SizeType N,
          typename Predicate>
constexpr FixedSizeTuple<NewT, SizeType, N>
map(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<NewT, SizeType, N> mapped_tuple(
      TuplePad<FixedSizeTuple<NewT, SizeType, N>>(tuple.size()));
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    mapped_tuple[i] = p(tuple[i]);
  }
  return mapped_tuple;
}

/**
 * Like map, except passes the index of each entry in tuple to the
 * given predicate, rather than the value.
 *
 * This is primarily useful when your predicate can capture some other
 * object that it needs to interact with.
 */
template <typename T,
          typename SizeType,
          SizeType N,
          typename Predicate>
constexpr FixedSizeTuple<T, SizeType, N>
map_index(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<T, SizeType, N> mapped_tuple(
      TuplePad<FixedSizeTuple<T, SizeType, N>>(tuple.size()));
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    mapped_tuple[i] = p(i);
  }
  return mapped_tuple;
}

/** map_index with a different FixedSizeTuple return type. */
template <typename NewT,
          typename T,
          typename SizeType,
          SizeType N,
          typename Predicate>
constexpr FixedSizeTuple<NewT, SizeType, N>
map_index(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<NewT, SizeType, N> mapped_tuple(
      TuplePad<FixedSizeTuple<NewT, SizeType, N>>(tuple.size()));
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    mapped_tuple[i] = p(i);
  }
  return mapped_tuple;
}

/**
 * Return a new tuple consisting only of the entries in the original
 * tuple where the predicate returns true.
 */
template <typename T, typename SizeType, SizeType N, typename Predicate>
constexpr FixedSizeTuple<T, SizeType, N>
filter(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<T, SizeType, N> filtered_tuple;
  for (const auto& v : tuple)
  {
    if (p(v))
    {
      filtered_tuple.append(v);
    }
  }
  return filtered_tuple;
}

/**
 * Like filter, except passes the index of each entry in tuple to the
 * given predicate, rather than the value.
 *
 * This is primarily useful when your predicate can capture some other
 * object that it needs to interact with.
 */
template <typename T, typename SizeType, SizeType N, typename Predicate>
constexpr FixedSizeTuple<T, SizeType, N>
filter_index(const FixedSizeTuple<T, SizeType, N>& tuple, Predicate p)
{
  FixedSizeTuple<T, SizeType, N> filtered_tuple;
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    if (p(i))
    {
      filtered_tuple.append(tuple[i]);
    }
  }
  return filtered_tuple;
}

/** @brief Get all but the last element of the input tuple */
template <typename T, typename SizeType, SizeType N>
constexpr FixedSizeTuple<T, SizeType, N>
init(FixedSizeTuple<T, SizeType, N> const& in) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(!in.is_empty(), "Cannot get init of empty tuple");
  FixedSizeTuple<T, SizeType, N> out{in};
  out.set_size(in.size() - 1);
  return out;
}

/** @brief Get the last element of the input tuple */
template <typename T, typename SizeType, SizeType N>
constexpr T last(FixedSizeTuple<T, SizeType, N> const& in) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(!in.is_empty(), "Cannot get last of empty tuple");
  return in[in.size() - 1];
}

/**
 * Return a tuple containing the first n elements of the given tuple.
 */
template <typename T, typename SizeType, SizeType N>
constexpr FixedSizeTuple<T, SizeType, N>
init_n(const FixedSizeTuple<T, SizeType, N>& tuple, const SizeType n) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(n <= tuple.size(),
                  "Cannot get more elements than present in tuple");
  FixedSizeTuple<T, SizeType, N> out{tuple};
  out.set_size(n);
  return out;
}

}  // namespace h2
