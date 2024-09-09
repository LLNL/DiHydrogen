////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <hip/hip_runtime.h>

namespace distconv
{
namespace tensor
{

inline hipStream_t get_gpu_stream(DefaultStream const& s)
{
  return 0;
}

inline hipStream_t get_gpu_stream(hipStream_t const& s)
{
  return s;
}

} // namespace tensor
} // namespace distconv
