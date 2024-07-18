////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <hip/hip_runtime.h>

#include "h2/utils/Error.hpp"


#define H2_CHECK_HIP(CMD)                                               \
  do                                                                    \
  {                                                                     \
    const auto status_H2_CHECK_HIP = (CMD);                             \
    if (status_H2_CHECK_HIP != hipSuccess)                              \
    {                                                                   \
      throw H2FatalException("HIP error ",                              \
                             ::h2::gpu::error_name(status_H2_CHECK_HIP), \
                             " (",                                      \
                             status_H2_CHECK_HIP,                       \
                             "): ",                                     \
                             ::h2::gpu::error_string(status_H2_CHECK_HIP)); \
    }                                                                   \
  } while (0)

namespace h2
{
namespace gpu
{

typedef hipStream_t DeviceStream;
typedef hipEvent_t DeviceEvent;
typedef hipError_t DeviceError;

inline bool ok(DeviceError status) noexcept
{
    return (status == hipSuccess);
}

inline char const* error_name(DeviceError status) noexcept
{
    return hipGetErrorName(status);
}

inline char const* error_string(DeviceError status) noexcept
{
    return hipGetErrorString(status);
}

} // namespace gpu
} // namespace h2
