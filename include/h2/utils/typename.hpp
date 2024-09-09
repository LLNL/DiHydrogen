////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Utilities for getting string representations of types.
 */

#include <string>
#include <typeinfo>

namespace h2
{

namespace internal
{
/**
 * Return a string name for a type, given its `type_info`.
 *
 * This will attempt to provide a readable name.
 */
std::string get_type_name(std::type_info const& tinfo);
}  // namespace internal

/** Return a string naming the given type. */
template <typename T>
inline std::string TypeName()
{
  return internal::get_type_name(typeid(T));
}

// Specializations for standard built-in types.
#define H2_ADD_TYPENAME(Type)                                                  \
  template <>                                                                  \
  inline std::string TypeName<Type>()                                          \
  {                                                                            \
    return #Type;                                                              \
  }

H2_ADD_TYPENAME(bool)
H2_ADD_TYPENAME(char)
H2_ADD_TYPENAME(unsigned char)
H2_ADD_TYPENAME(signed char)
H2_ADD_TYPENAME(short)
H2_ADD_TYPENAME(unsigned short)
H2_ADD_TYPENAME(int)
H2_ADD_TYPENAME(unsigned int)
H2_ADD_TYPENAME(long)
H2_ADD_TYPENAME(unsigned long)
H2_ADD_TYPENAME(long long)
H2_ADD_TYPENAME(unsigned long long)
H2_ADD_TYPENAME(float)
H2_ADD_TYPENAME(double)
H2_ADD_TYPENAME(long double)
// TODO: fp16, bf16, etc. when we have them.

#undef H2_ADD_TYPENAME

}  // namespace h2
