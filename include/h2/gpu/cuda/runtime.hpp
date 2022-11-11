////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once
#ifndef H2_INCLUDE_H2_GPU_RUNTIME_CUDA_HPP_INCLUDED
#define H2_INCLUDE_H2_GPU_RUNTIME_CUDA_HPP_INCLUDED

#include <cuda_runtime.h>

#define H2_CHECK_CUDA(CMD)                                                     \
    do                                                                         \
    {                                                                          \
        if (cudaSuccess != (CMD))                                              \
        {                                                                      \
            throw std::runtime_error("Something bad happened.");               \
        }                                                                      \
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
#endif // H2_INCLUDE_H2_GPU_RUNTIME_CUDA_HPP_INCLUDED
