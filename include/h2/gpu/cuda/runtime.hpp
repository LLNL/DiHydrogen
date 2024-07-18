////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cuda_runtime.h>

#include "h2/utils/Error.hpp"


#define H2_CHECK_CUDA(CMD)                                              \
  do                                                                    \
  {                                                                     \
    const auto status_H2_CHECK_CUDA = (CMD);                            \
    if (status_H2_CHECK_CUDA != cudaSuccess)                            \
    {                                                                   \
      throw H2FatalException("CUDA error ",                             \
                             ::h2::gpu::error_name(status_H2_CHECK_CUDA), \
                             " (",                                      \
                             status_H2_CHECK_CUDA,                      \
                             "): ",                                     \
                             ::h2::gpu::error_string(status_H2_CHECK_CUDA)); \
    }                                                                   \
  } while (0)

namespace h2
{
namespace gpu
{

typedef cudaStream_t DeviceStream;
typedef cudaEvent_t DeviceEvent;
typedef cudaError_t DeviceError;

inline bool ok(DeviceError status) noexcept
{
    return (status == cudaSuccess);
}

inline char const* error_name(DeviceError status) noexcept
{
    return cudaGetErrorName(status);
}

inline char const* error_string(DeviceError status) noexcept
{
    return cudaGetErrorString(status);
}

} // namespace gpu
} // namespace h2
