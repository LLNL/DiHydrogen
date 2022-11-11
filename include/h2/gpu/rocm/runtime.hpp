////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once
#ifndef H2_INCLUDE_H2_GPU_ROCM_RUNTIME_HPP_INCLUDED
#define H2_INCLUDE_H2_GPU_ROCM_RUNTIME_HPP_INCLUDED

#include <hip/hip_runtime.h>

#define H2_CHECK_HIP(CMD)                                                      \
    do                                                                         \
    {                                                                          \
        auto const hip_status_h2_check_hip = (CMD);                            \
        if (hipSuccess != hip_status_h2_check_hip)                             \
        {                                                                      \
            std::ostringstream oss;                                            \
            oss << __FILE__ << ":" << __LINE__ << ": "                         \
                << "HIP runtime command failed:\n\n  " #CMD "\n\n"             \
                << "Error name: "                                              \
                << ::h2::gpu::error_name(hip_status_h2_check_hip)              \
                << "\n Error msg: "                                            \
                << ::h2::gpu::error_string(hip_status_h2_check_hip) << "\n";   \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
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
#endif // H2_INCLUDE_H2_GPU_ROCM_RUNTIME_HPP_INCLUDED
