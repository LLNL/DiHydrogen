////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2_config.hpp>
#include <h2/utils/Error.hpp>

#include <execinfo.h>
#include <dlfcn.h>

#include <iomanip>
#include <memory>
#include <sstream>

#if __has_include(<cxxabi.h>)
#include <cxxabi.h>
#define H2_HAS_CXXABI_H
#endif

#ifndef H2_DEBUG
// Only used when not in debug mode.
#include "h2/utils/environment_vars.hpp"
#endif

bool H2ExceptionBase::should_save_backtrace() const
{
#ifdef H2_DEBUG
  return true;  // Always save backtraces in debug mode.
#else
  // Save if H2_DEBUG_BACKTRACE is set.
  return h2::env::get<bool>("DEBUG_BACKTRACE");
#endif
}

void H2ExceptionBase::collect_backtrace()
{
  constexpr int max_frames = 128;
  using c_str_ptr = std::unique_ptr<char, void (*)(void*)>;
  using c_str_ptr_ptr = std::unique_ptr<char*, void (*)(void*)>;

  void* frames[max_frames];
  const int num_frames = backtrace(frames, max_frames);

  c_str_ptr_ptr symbols{backtrace_symbols(frames, num_frames), free};

  // This deliberately does not reuse machinery from `safe_demangle` to
  // avoid any dependencies.
  // Note we cannot directly demangle the entries returned by
  // backtrace_symbols.
  std::ostringstream ss;
  ss << what_ << "\nStack trace:\n";
  for (int i = 0; i < num_frames; ++i)
  {
    ss << std::setw(4) << i << ": ";
#ifdef H2_HAS_CXXABI_H
    Dl_info info;
    dladdr(frames[i], &info);
    if (info.dli_sname != nullptr)
    {
      c_str_ptr demangled{abi::__cxa_demangle(
          info.dli_sname, nullptr, nullptr, nullptr), free};
      if (demangled)
      {
        ss << demangled.get();
      }
      else
      {
        ss << info.dli_sname << " (demangling failed)";
      }
    }
    else
#endif  // H2_HAS_CXXABI_H
    {
      if (symbols)
      {
        ss << symbols.get()[i] << " ";
      }
      ss << "(could not find stack frame symbol)";
    }
    ss << "\n";
  }

  what_ = ss.str();
}

namespace h2
{

void break_on_me(std::string const& msg)
{
    char const volatile* x = msg.data();
    (void) x;
}

} // namespace h2
