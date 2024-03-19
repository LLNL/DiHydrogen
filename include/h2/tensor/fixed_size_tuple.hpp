////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <iterator>
#include <type_traits>
#include <ostream>

#include "h2/utils/Error.hpp"

/** @file
 *
 * Defines fixed-size tuples and associated functions.
 */

namespace h2 {

/** Helper struct for initializing a FixedSizeTuple with padding. */
template <typename T, typename SizeType>
struct FixedSizeTuplePadding {
  SizeType size_;  /**< Number of valid elements. */
  T pad_value_;  /**< Value to set unspecified valid elements to. */

  constexpr FixedSizeTuplePadding(SizeType size, T pad_value) :
    size_(size), pad_value_(pad_value) {}
};

/** Helper for constructing tuple paddings. */
template <typename TupleType>
inline constexpr FixedSizeTuplePadding<typename TupleType::type, typename TupleType::size_type>
TuplePad(typename TupleType::size_type size, typename TupleType::type pad_value = {}) {
  return FixedSizeTuplePadding<typename TupleType::type, typename TupleType::size_type>(size, pad_value);
}

/** Represent a tuple with a constant maximum size. */
template <typename T, typename SizeType, SizeType N>
struct FixedSizeTuple {
  std::array<T, N> data_;  /**< Fixed size data buffer. */
  SizeType size_; /**< Number of valid elements in the tuple. */

  using type = T;
  using size_type = SizeType;
  static constexpr SizeType max_size = N;

  using iterator = typename std::array<T, N>::iterator;
  using const_iterator = typename std::array<T, N>::const_iterator;
  using reverse_iterator = typename std::array<T, N>::reverse_iterator;
  using const_reverse_iterator = typename std::array<T, N>::const_reverse_iterator;

  /**
   * Construct a tuple from the arguments and set the number of valid
   * elements based on the number of arguments.
   */
  template <typename... Args>
  constexpr FixedSizeTuple(Args... args) : data_{args...}, size_(sizeof...(args)) {}

  /**
   * Construct a tuple with a specified number of valid elements and
   * possibly some specified entries, padding the remainder.
   */
  template <typename... Args>
  constexpr FixedSizeTuple(FixedSizeTuplePadding<T, SizeType> pad_arg, Args... args) :
    data_{args...}, size_(pad_arg.size_) {
    // Note: This does not check if sizeof...(args) > pad_arg.size_.
    // While that won't cause issues, it might indicate a correctness
    // problem in the caller's code.
    for (std::size_t i = sizeof...(args); i < pad_arg.size_; ++i) {
      data_[i] = pad_arg.pad_value_;
    }
  }

  FixedSizeTuple(const FixedSizeTuple& other) = default;
  FixedSizeTuple(FixedSizeTuple&& other) = default;
  FixedSizeTuple& operator=(const FixedSizeTuple& other) = default;
  FixedSizeTuple& operator=(FixedSizeTuple&& other) = default;

  /** Return the number of valid elements in the tuple. */
  constexpr SizeType size() const H2_NOEXCEPT { return size_; }

  /** Return true when there are no valid elements in the tuple. */
  constexpr bool empty() const H2_NOEXCEPT { return size_ == 0; }

  /** Return a raw pointer to the tuple. */
  T* data() H2_NOEXCEPT { return data_.data(); }

  const T* data() const H2_NOEXCEPT { return data_.data(); }

  /** Return a constant raw pointer to the tuple. */
  const T* const_data() const H2_NOEXCEPT { return data_.data(); }

  /** Return an iterator to the first element of the tuple. */
  constexpr iterator begin() H2_NOEXCEPT { return data_.begin(); }

  constexpr const_iterator begin() const H2_NOEXCEPT { return data_.begin(); }

  /** Return a constant iterator to the first element of the tuple. */
  constexpr const_iterator cbegin() const H2_NOEXCEPT { return data_.cbegin(); }

  /**
   * Return an iterator to the element past the last element of the
   * tuple.
   */
  constexpr iterator end() H2_NOEXCEPT { return data_.begin() + size_; }

  constexpr const_iterator end() const H2_NOEXCEPT
  {
    return data_.begin() + size_;
  }

  /**
   * Return a constant iterator to the element past the last element of
   * the tuple.
   */
  constexpr const_iterator cend() const H2_NOEXCEPT
  {
    return data_.cbegin() + size_;
  }

  /**
   * Return a reverse iterator to the first element of the reversed
   * tuple.
   */
  constexpr reverse_iterator rbegin() H2_NOEXCEPT
  {
    return std::make_reverse_iterator(end());
  }

  constexpr const_reverse_iterator rbegin() const H2_NOEXCEPT
  {
    return std::make_reverse_iterator(end());
  }

  /**
   * Return a const reverse iterator to the first element of the
   * reversed tuple.
   */
  constexpr const_reverse_iterator crbegin() const H2_NOEXCEPT
  {
    return std::make_reverse_iterator(cend());
  }

  /**
   * Return a reverse iterator to the element past the last element of
   * the reversed tuple.
   */
  constexpr reverse_iterator rend() H2_NOEXCEPT
  {
    return std::make_reverse_iterator(begin());
  }

  constexpr const_reverse_iterator rend() const H2_NOEXCEPT
  {
    return std::make_reverse_iterator(begin());
  }

  /**
   * Return a const reverse iterator to the element past the last
   * element of the reversed tuple.
   */
  constexpr const_reverse_iterator crend() const H2_NOEXCEPT
  {
    return std::make_reverse_iterator(cbegin());
  }

  /** Return the value of the tuple at the i'th index. */
  constexpr T& operator[](SizeType i) H2_NOEXCEPT {
    H2_ASSERT_DEBUG(i < size_, "Tuple index too large");
    return data_[i];
  }

  /** Return the value of the tuple at the i'th index. */
  constexpr const T& operator[](SizeType i) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(i < size_, "Tuple index too large");
    return data_[i];
  }

  /** Set the entry at the i'th index to v. */
  constexpr void set(SizeType i, T v) H2_NOEXCEPT {
    H2_ASSERT_DEBUG(i < size_, "Tuple index too large");
    data_[i] = v;
  }

  /**
   * Change the number of valid elements in the tuple.
   *
   * If the new size is larger than the existing size, the value of
   * newly-valid entries is undefined until they are set.
   */
  constexpr void set_size(SizeType new_size) H2_NOEXCEPT {
    H2_ASSERT_DEBUG(new_size <= N, "New size exceeds max");
    size_ = new_size;
  }

  /**
   * Compare two tuples for equaltiy.
   *
   * Tuples are equal if they have the same size and all valid elements
   * are equal. Note they do not need to have the same maximum size or
   * type; the types just need to be comparable.
   */
  template <typename U, typename OtherSizeType, OtherSizeType M>
  constexpr bool operator==(const FixedSizeTuple<U, OtherSizeType, M>& other) const H2_NOEXCEPT {
    if (size_ != other.size_) {
      return false;
    }
    for (SizeType i = 0; i < size_; ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * Compare two tuples for inequality.
   */
  template <typename U, typename OtherSizeType, OtherSizeType M>
  constexpr bool operator!=(const FixedSizeTuple<U, OtherSizeType, M>& other) const H2_NOEXCEPT {
    if (size_ != other.size_) {
      return true;
    }
    for (SizeType i = 0; i < size_; ++i) {
      if (data_[i] != other.data_[i]) {
        return true;
      }
    }
    return false;
  }
};

/** Operator overload for printing tuples (when the type is printable). */
template <typename T, typename SizeType, SizeType N>
inline std::ostream& operator<<(std::ostream& os, const FixedSizeTuple<T, SizeType, N>& tuple) {
  os << "{";
  for (SizeType i = 0; i < tuple.size(); ++i) {
    os << tuple[i];
    // Note: tuple.size() must be >= 1 for us to be here.
    if (i < tuple.size() - 1) {
      os << ", ";
    }
  }
  os << "}";
  return os;
}

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
  for (SizeType i = 0; i < tuple.size(); ++i) {
    if (p(tuple[i])) {
      return true;
    }
  }
  return false;
}

/** @brief Get all but the last element of the input tuple */
template <typename T, typename SizeType, SizeType N>
constexpr FixedSizeTuple<T, SizeType, N>
init(FixedSizeTuple<T, SizeType, N> const& in)
{
  if (in.empty())
    throw std::runtime_error("cannot get init of empty FixedSizeTuple");
  FixedSizeTuple<T, SizeType, N> out{in};
  out.set_size(in.size() - 1);
  return out;
}

/** @brief Get the last element of the input tuple */
template <typename T, typename SizeType, SizeType N>
constexpr T last(FixedSizeTuple<T, SizeType, N> const& in)
{
  if (in.empty())
    throw std::runtime_error("cannot get last of empty FixedSizeTuple");
  return in[in.size() - 1];
}

/**
 * Return a tuple containing the first n elements of the given tuple.
 */
template <typename T, typename SizeType, SizeType N>
constexpr FixedSizeTuple<T, SizeType, N>
init_n(const FixedSizeTuple<T, SizeType, N>& tuple, const SizeType n)
{
  H2_ASSERT_DEBUG(n <= tuple.size(),
                  "Cannot get more elements than present in tuple");
  FixedSizeTuple<T, SizeType, N> out{tuple};
  out.set_size(n);
  return out;
}

}  // namespace h2

namespace std
{

// Inject hash specializations for tuples.

template <typename T, typename SizeType, SizeType N>
struct hash<h2::FixedSizeTuple<T, SizeType, N>>
{
  size_t
  operator()(const h2::FixedSizeTuple<T, SizeType, N>& tuple) const noexcept
  {
    // Mixing adapted from Boost.
    // Hash both the size and elements.
    size_t seed = hash<SizeType>()(tuple.size());
    auto hasher = hash<T>();
    for (SizeType i = 0; i < tuple.size(); ++i)
    {
      seed ^= hasher(tuple[i]) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

}  // namespace std
