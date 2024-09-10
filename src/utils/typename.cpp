////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/utils/typename.hpp"

#include "h2/utils/Error.hpp"

#include <memory>
#include <unordered_map>

#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define H2_HAS_CXXABI_H
#endif

#ifdef H2_HAS_CXXABI_H

// Demangle mangled, ensuring no memory leaks.
std::string safe_demangle(char const* mangled)
{
  H2_ASSERT_ALWAYS(mangled != nullptr, "Attempt to demangle a null pointer");

  using c_str_ptr = std::unique_ptr<char, void (*)(void*)>;
  static std::unordered_map<int, char const*> const demangle_errors = {
    {-1, "Memory allocation failure"},
    {-2, "Mangled name is invalid"},
    {-3, "Invalid arguments"}};

  int status;
  c_str_ptr demangled{abi::__cxa_demangle(mangled, nullptr, nullptr, &status),
                      free};

  if (status != 0)
  {
    auto error_str = demangle_errors.find(status);
    if (error_str != demangle_errors.end())
    {
      throw H2NonfatalException(std::string("Demangling failed: ")
                                + error_str->second);
    }
    else
    {
      throw H2NonfatalException("Demangling failed: Unknown error");
    }
  }

  return demangled.get();
}

#endif  // H2_HAS_CXXABI_H

namespace h2
{

namespace internal
{
std::string get_type_name(std::type_info const& tinfo)
{
#ifdef H2_HAS_CXXABI_H
  try
  {
    return safe_demangle(tinfo.name());
  }
  catch (H2NonfatalException const&)
  {
    return tinfo.name();  // Getting a type name should not kill us.
  }
#else
  return tinfo.name();
#endif
}

}  // namespace internal

}  // namespace h2
