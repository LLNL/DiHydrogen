////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/utils/Error.hpp"

#include <array>
#include <iterator>
#include <ostream>
#include <type_traits>

/** @file
 *
 * Defines fixed-size tuples.
 */

namespace h2
{

/** Helper struct for initializing a FixedSizeTuple with padding. */
template <typename T, typename SizeType>
struct FixedSizeTuplePadding
{
  SizeType size_; /**< Number of valid elements. */
  T pad_value_;   /**< Value to set unspecified valid elements to. */

  constexpr FixedSizeTuplePadding(SizeType size, T pad_value)
    : size_(size), pad_value_(pad_value)
  {}
};

/** Helper for constructing tuple paddings. */
template <typename TupleType>
inline constexpr FixedSizeTuplePadding<typename TupleType::type,
                                       typename TupleType::size_type>
TuplePad(typename TupleType::size_type size,
         typename TupleType::type pad_value = {})
{
  return FixedSizeTuplePadding<typename TupleType::type,
                               typename TupleType::size_type>(size, pad_value);
}

/** Represent a tuple with a constant maximum size. */
template <typename T, typename SizeType, SizeType N>
struct FixedSizeTuple
{
  std::array<T, N> data_; /**< Fixed size data buffer. */
  SizeType size_;         /**< Number of valid elements in the tuple. */

  using type = T;
  using size_type = SizeType;
  static constexpr SizeType max_size = N;

  using iterator = typename std::array<T, N>::iterator;
  using const_iterator = typename std::array<T, N>::const_iterator;
  using reverse_iterator = typename std::array<T, N>::reverse_iterator;
  using const_reverse_iterator =
    typename std::array<T, N>::const_reverse_iterator;

  /**
   * Construct a tuple from the arguments and set the number of valid
   * elements based on the number of arguments.
   */
  template <typename... Args>
  constexpr FixedSizeTuple(Args... args)
    : data_{args...}, size_(sizeof...(args))
  {}

  /**
   * Construct a tuple with a specified number of valid elements and
   * possibly some specified entries, padding the remainder.
   */
  template <typename... Args>
  constexpr FixedSizeTuple(FixedSizeTuplePadding<T, SizeType> pad_arg,
                           Args... args)
    : data_{args...}, size_(pad_arg.size_)
  {
    // Note: This does not check if sizeof...(args) > pad_arg.size_.
    // While that won't cause issues, it might indicate a correctness
    // problem in the caller's code.
    for (SizeType i = static_cast<SizeType>(sizeof...(args)); i < pad_arg.size_;
         ++i)
    {
      data_[i] = pad_arg.pad_value_;
    }
  }

  FixedSizeTuple(FixedSizeTuple const& other) = default;
  FixedSizeTuple(FixedSizeTuple&& other) = default;
  FixedSizeTuple& operator=(FixedSizeTuple const& other) = default;
  FixedSizeTuple& operator=(FixedSizeTuple&& other) = default;

  /**
   * Construct a tuple from another tuple with data type that is
   * convertible to T.
   */
  template <typename U, typename OtherSizeType, OtherSizeType M>
  static constexpr FixedSizeTuple
  convert_from(FixedSizeTuple<U, OtherSizeType, M> const& other)
  {
    static_assert(
      std::is_convertible_v<U, T>,
      "Cannot construct a tuple from another with unconvertible type");
    static_assert(N >= M,
                  "Cannot construct a tuple from another that may be larger");
    FixedSizeTuple new_tuple;
    for (SizeType i = 0; i < other.size_; ++i)
    {
      new_tuple.append(other[i]);
    }
    return new_tuple;
  }

  /** Return the number of valid elements in the tuple. */
  constexpr SizeType size() const H2_NOEXCEPT { return size_; }

  /** Return true when there are no valid elements in the tuple. */
  constexpr bool is_empty() const H2_NOEXCEPT { return size_ == 0; }

  /** Return a raw pointer to the tuple. */
  T* data() H2_NOEXCEPT { return data_.data(); }

  T const* data() const H2_NOEXCEPT { return data_.data(); }

  /** Return a constant raw pointer to the tuple. */
  T const* const_data() const H2_NOEXCEPT { return data_.data(); }

  /** Return a reference to the first element in the tuple. */
  constexpr T& front() H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(size_ > 0, "Cannot access front in empty tuple");
    return data_[0];
  }

  constexpr T const& front() const H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(size_ > 0, "Cannot access front in empty tuple");
    return data_[0];
  }

  /** Return a reference to the last element in the tuple. */
  constexpr T& back() H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(size_ > 0, "Cannot access back in empty tuple");
    return data_[size_ - 1];
  }

  constexpr T const& back() const H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(size_ > 0, "Cannot access back in empty tuple");
    return data_[size_ - 1];
  }

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
  constexpr T& operator[](SizeType i) H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(i < size_,
                    "Tuple index ",
                    i,
                    " too large (must be less than ",
                    size_,
                    ")");
    return data_[i];
  }

  /** Return the value of the tuple at the i'th index. */
  constexpr T const& operator[](SizeType i) const H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(i < size_,
                    "Tuple index ",
                    i,
                    " too large (must be less than ",
                    size_,
                    ")");
    return data_[i];
  }

  /** Set the entry at the i'th index to v. */
  constexpr void set(SizeType i, T v) H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(i < size_,
                    "Tuple index ",
                    i,
                    " too large (must be less than ",
                    size_,
                    ")");
    data_[i] = v;
  }

  /**
   * Change the number of valid elements in the tuple.
   *
   * If the new size is larger than the existing size, the value of
   * newly-valid entries is undefined until they are set.
   */
  constexpr void set_size(SizeType new_size) H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(
      new_size <= N, "New size ", new_size, " exceeds max size ", N);
    size_ = new_size;
  }

  /**
   * Add a value to the end of the tuple and increase the number of
   * valid entries by one.
   *
   * It is an error to call this if this would exceed the maximum size
   * of the tuple.
   */
  constexpr void append(T v) H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(size_ < N - 1, "Append would exceed tuple size ", N);
    data_[size_] = v;
    ++size_;
  }

  /**
   * Compare two tuples for equaltiy.
   *
   * Tuples are equal if they have the same size and all valid elements
   * are equal. Note they do not need to have the same maximum size or
   * type; the types just need to be comparable.
   */
  template <typename U, typename OtherSizeType, OtherSizeType M>
  constexpr bool
  operator==(FixedSizeTuple<U, OtherSizeType, M> const& other) const H2_NOEXCEPT
  {
    if (size_ != other.size_)
    {
      return false;
    }
    for (SizeType i = 0; i < size_; ++i)
    {
      if (data_[i] != other.data_[i])
      {
        return false;
      }
    }
    return true;
  }

  /**
   * Compare two tuples for inequality.
   */
  template <typename U, typename OtherSizeType, OtherSizeType M>
  constexpr bool
  operator!=(FixedSizeTuple<U, OtherSizeType, M> const& other) const H2_NOEXCEPT
  {
    if (size_ != other.size_)
    {
      return true;
    }
    for (SizeType i = 0; i < size_; ++i)
    {
      if (data_[i] != other.data_[i])
      {
        return true;
      }
    }
    return false;
  }
};

/** Print a tuple with some support for customization. */
template <typename T, typename SizeType, SizeType N>
inline void print_tuple(std::ostream& os,
                        FixedSizeTuple<T, SizeType, N> const& tuple,
                        std::string const start_brace = "{",
                        std::string const end_brace = "}",
                        std::string const separator = ", ")
{
  os << start_brace;
  for (SizeType i = 0; i < tuple.size(); ++i)
  {
    os << tuple[i];
    // Note: tuple.size() must be >= 1 for us to be here.
    if (i < tuple.size() - 1)
    {
      os << separator;
    }
  }
  os << end_brace;
}

/** Operator overload for printing tuples (when the type is printable). */
template <typename T, typename SizeType, SizeType N>
inline std::ostream& operator<<(std::ostream& os,
                                FixedSizeTuple<T, SizeType, N> const& tuple)
{
  print_tuple(os, tuple);
  return os;
}

}  // namespace h2

namespace std
{

// Inject hash specializations for tuples.

template <typename T, typename SizeType, SizeType N>
struct hash<h2::FixedSizeTuple<T, SizeType, N>>
{
  size_t
  operator()(h2::FixedSizeTuple<T, SizeType, N> const& tuple) const noexcept
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
