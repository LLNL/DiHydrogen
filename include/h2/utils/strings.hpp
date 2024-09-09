////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Utilities for working with strings.
 */

#include "h2/utils/As.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace h2
{

/**
 * Build a string by concatenating all the arguments to this function.
 *
 * All arguments must be stream-outputable (i.e., have an operator<<
 * defined fo the type).
 *
 * @tparam Args The (inferred) types of the arguments.
 *
 * @param[in] args The things to be concatenated into a string.
 */
template <typename... Args>
inline std::string build_string(Args&&... args) noexcept(sizeof...(Args) == 0)
{
  if constexpr (sizeof...(Args) == 0)
  {
    return std::string();
  }
  else
  {
    std::ostringstream ss;
    (ss << ... << args);
    return ss.str();
  }
}

inline std::string build_string(const char* s)
{
  return std::string(s);
}

inline std::string build_string(const std::string& s)
{
  return std::string(s);
}

/** Return an upper-cased version of a string. */
inline std::string str_toupper(std::string str)
{
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return std::toupper(c);
  });
  return str;
}

/** Return a lower-cased version of a string. */
inline std::string str_tolower(std::string str)
{
  std::transform(str.begin(), str.end(), str.begin(), [](unsigned char c) {
    return std::tolower(c);
  });
  return str;
}

/**
 * Convert an input string to type T.
 */
template <typename T>
inline T from_string(const std::string& str);

template <>
inline unsigned int from_string<unsigned int>(const std::string&);

inline std::string from_string(std::string&& str)
{
  return std::move(str);
}

template <>
inline std::string from_string<std::string>(const std::string& str)
{
  return str;
}

template <>
inline int from_string<int>(const std::string& str)
{
  return std::stoi(str);
}

template <>
inline long from_string<long>(const std::string& str)
{
  return std::stol(str);
}

template <>
inline long long from_string<long long>(const std::string& str)
{
  return std::stoll(str);
}

template <>
inline char from_string<char>(const std::string& str)
{
  if constexpr (std::numeric_limits<char>::is_signed)
  {
    int tmp = from_string<int>(str);
    if (tmp > std::numeric_limits<char>::max())
    {
      throw std::out_of_range("from_string: too large for char");
    }
    if (tmp < std::numeric_limits<char>::min())
    {
      throw std::out_of_range("from_string: too small for char");
    }
    return as<char>(tmp);
  }
  else
  {
    unsigned int tmp = from_string<unsigned int>(str);
    if (tmp > std::numeric_limits<char>::max())
    {
      throw std::out_of_range("from_string: too large for char");
    }
    return as<char>(tmp);
  }
}

template <>
inline signed char from_string<signed char>(const std::string& str)
{
  int tmp = from_string<int>(str);
  if (tmp > std::numeric_limits<signed char>::max())
  {
    throw std::out_of_range("from_string: too large for signed char");
  }
  if (tmp < std::numeric_limits<signed char>::min())
  {
    throw std::out_of_range("from_string: too small for signed char");
  }
  return as<signed char>(tmp);
}

template <>
inline short from_string<short>(const std::string& str)
{
  int tmp = from_string<int>(str);
  if (tmp > std::numeric_limits<short>::max())
  {
    throw std::out_of_range("from_string: too large for short");
  }
  if (tmp < std::numeric_limits<short>::min())
  {
    throw std::out_of_range("from_string: too small for short");
  }
  return as<short>(tmp);
}

template <>
inline unsigned long from_string<unsigned long>(const std::string& str)
{
  return std::stoul(str);
}

template <>
inline unsigned long long
from_string<unsigned long long>(const std::string& str)
{
  return std::stoull(str);
}

// TODO: Conversion of negative numbers ("-42") to these types may not
// work correctly.

template <>
inline unsigned char from_string<unsigned char>(const std::string& str)
{
  unsigned long tmp = from_string<unsigned long>(str);
  if (tmp > std::numeric_limits<unsigned char>::max())
  {
    throw std::out_of_range("from_string: too large for unsigned char");
  }
  return as<unsigned char>(tmp);
}

template <>
inline unsigned short from_string<unsigned short>(const std::string& str)
{
  unsigned long tmp = from_string<unsigned long>(str);
  if (tmp > std::numeric_limits<unsigned short>::max())
  {
    throw std::out_of_range("from_string: too large for unsigned short");
  }
  return as<unsigned short>(tmp);
}

template <>
inline unsigned int from_string<unsigned int>(const std::string& str)
{
  unsigned long tmp = from_string<unsigned long>(str);
  if (tmp > std::numeric_limits<unsigned int>::max())
  {
    throw std::out_of_range("from_string: too large for unsigned int");
  }
  return as<unsigned int>(tmp);
}

template <>
inline float from_string<float>(const std::string& str)
{
  return std::stof(str);
}

template <>
inline double from_string<double>(const std::string& str)
{
  return std::stod(str);
}

template <>
inline long double from_string<long double>(const std::string& str)
{
  return std::stold(str);
}

/**
 * Convert a string to a boolean.
 *
 * Values considered true: the string "true" (any case); a string
 * convertible to an integer that is non-zero.
 *
 * Values considered false: the string "false" (any case); a string
 * convertible to an integer that is zero.
 *
 * An empty string will throw. Other strings will throw.
 */
template <>
inline bool from_string<bool>(const std::string& str)
{
  std::string upstr = str_toupper(str);
  if (upstr == "TRUE")
  {
    return true;
  }
  else if (upstr == "FALSE")
  {
    return false;
  }
  else
  {
    return as<bool>(from_string<long long>(str));
  }
}

} // namespace h2
