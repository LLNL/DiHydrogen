////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/core/dispatch.hpp"

#include <unordered_map>

namespace h2
{
namespace internal
{

namespace
{

using KeyToEntryMap = std::unordered_map<DispatchKeyT, DispatchFunctionEntry>;
// Dispatch table with dynamic registration.
// Maps from names to a lookup table of dispatch keys -> dispatch entries.
std::unordered_map<std::string, KeyToEntryMap> dispatch_table;

}  // anonymous namespace

void add_dispatch_entry(std::string const& name,
                        DispatchKeyT const& dispatch_key,
                        DispatchFunctionEntry const& dispatch_entry)
{
  if (dispatch_table.count(name) == 0)
  {
    dispatch_table.emplace(name, KeyToEntryMap{});
  }
  dispatch_table[name][dispatch_key] = dispatch_entry;
}

bool has_dispatch_entry(std::string const& name,
                        DispatchKeyT const& dispatch_key)
{
  return (dispatch_table.count(name) > 0)
         && (dispatch_table[name].count(dispatch_key) > 0);
}

DispatchFunctionEntry const&
get_dispatch_entry(std::string const& name, DispatchKeyT const& dispatch_key)
{
  if (!has_dispatch_entry(name, dispatch_key))
  {
    throw H2FatalException("Attempt to look up dispatch for name ",
                           name,
                           " and key ",
                           dispatch_key,
                           " which does not exist");
  }
  return dispatch_table[name][dispatch_key];
}

}  // namespace internal

void dispatch_unregister(std::string const& name,
                         internal::DispatchKeyT const& dispatch_key)
{
  if (internal::has_dispatch_entry(name, dispatch_key))
  {
    internal::dispatch_table[name].erase(dispatch_key);
  }
}

}  // namespace h2

// *****
// Static dispatch example (also used in unit testing).

namespace h2
{

namespace impl
{

template <typename T>
void dispatch_test_impl(CPUDev_t, T* v)
{
  *v = 42;
}

#ifdef H2_HAS_GPU
template <typename T>
void dispatch_test_impl(GPUDev_t, T* v)
{}
#endif

// Instantiate for all compute types:
#define PROTO(device, t1) template void dispatch_test_impl<t1>(device, t1*)
H2_INSTANTIATE_1
#undef PROTO

}  // namespace impl

}  // namespace h2

// End static dispatch example.
// *****
