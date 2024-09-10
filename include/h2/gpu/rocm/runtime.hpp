////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/utils/Error.hpp"

#include <hip/hip_runtime.h>

#define H2_CHECK_HIP(CMD)                                                      \
  do                                                                           \
  {                                                                            \
    const auto status_H2_CHECK_HIP = (CMD);                                    \
    if (status_H2_CHECK_HIP != hipSuccess)                                     \
    {                                                                          \
      throw H2FatalException("HIP error ",                                     \
                             ::h2::gpu::error_name(status_H2_CHECK_HIP),       \
                             " (",                                             \
                             status_H2_CHECK_HIP,                              \
                             "): ",                                            \
                             ::h2::gpu::error_string(status_H2_CHECK_HIP));    \
    }                                                                          \
  } while (0)

namespace h2
{
namespace gpu
{

typedef hipStream_t DeviceStream;
typedef hipEvent_t DeviceEvent;
typedef hipError_t DeviceError;

constexpr unsigned int max_grid_x = 2147483647;
constexpr unsigned int max_grid_y = 65536;
constexpr unsigned int max_grid_z = 65536;
constexpr unsigned int max_block_x = 1024;
constexpr unsigned int max_block_y = 1024;
constexpr unsigned int max_block_z = 1024;
constexpr unsigned int max_threads_per_block = 1024;
constexpr unsigned int warp_size = 64;

// Useful defaults for decomposing work:
constexpr unsigned int num_threads_per_block = 256;
constexpr unsigned int work_per_thread = 4;
constexpr unsigned int work_per_block = work_per_thread * num_threads_per_block;

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

template <typename... KernelArgs, typename... Args>
void launch_kernel_internal(void (*kernel)(KernelArgs...),
                            dim3 const& grid_dim,
                            dim3 const& block_dim,
                            std::size_t shared_mem,
                            DeviceStream stream,
                            Args&&... args)
{
  // Assumes Args and KernelArgs have been checked.
  H2_CHECK_HIP(hipGetLastError());
  hipLaunchKernelGGL(kernel,
                     grid_dim,
                     block_dim,
                     shared_mem,
                     stream,
                     std::forward<Args>(args)...);
  H2_CHECK_HIP(hipGetLastError());
}

}  // namespace gpu
}  // namespace h2
