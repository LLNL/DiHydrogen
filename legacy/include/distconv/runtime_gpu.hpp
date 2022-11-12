////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once
#ifndef H2_LEGACY_INCLUDE_DISTCONV_RUNTIME_GPU_HPP_INCLUDED
#define H2_LEGACY_INCLUDE_DISTCONV_RUNTIME_GPU_HPP_INCLUDED

#include "distconv/util/util_gpu.hpp"
#include "distconv_config.hpp"
#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#if H2_HAS_CUDA

#include "runtime_cuda.hpp"

#define GPU_DEVICE_RESET cudaDeviceReset
#define GPU_FREE cudaFree
#define GPU_GET_LAST_ERROR cudaGetLastError
#define GPU_MAKE_GPU_EXTENT make_cudaExtent
#define GPU_MAKE_GPU_PITCHED_PTR make_cudaPitchedPtr
#define GPU_MAKE_GPU_POS make_cudaPos
#define GPU_MALLOC cudaMalloc

// These aren't general-purpose; maybe best left in this weird
// preprocessor (anti)pattern.
#define GPU_MEMCPY_3D_ASYNC cudaMemcpy3DAsync
#define GPU_MEMCPY_DEFAULT cudaMemcpyDefault

#define DISTCONV_GPU_MALLOC(...) DISTCONV_CUDA_MALLOC(__VA_ARGS__)

#elif H2_HAS_ROCM

#include "runtime_rocm.hpp"

#define GPU_DEVICE_RESET hipDeviceReset
#define GPU_FREE hipFree
#define GPU_GET_LAST_ERROR hipGetLastError
#define GPU_MAKE_GPU_EXTENT make_hipExtent
#define GPU_MAKE_GPU_PITCHED_PTR make_hipPitchedPtr
#define GPU_MAKE_GPU_POS make_hipPos
#define GPU_MALLOC hipMalloc

// These aren't general-purpose; maybe best left in this weird
// preprocessor (anti)pattern.
#define GPU_MEMCPY_3D_ASYNC hipMemcpy3DAsync
#define GPU_MEMCPY_DEFAULT hipMemcpyDefault

#define DISTCONV_GPU_MALLOC(...) DISTCONV_HIP_MALLOC(__VA_ARGS__)

#endif

namespace distconv
{
namespace internal
{

#if H2_HAS_CUDA
using RuntimeGPU = RuntimeCUDA;
#elif H2_HAS_ROCM
using RuntimeGPU = RuntimeHIP;
#endif

} // namespace internal
} // namespace distconv
#endif // H2_LEGACY_INCLUDE_DISTCONV_RUNTIME_GPU_HPP_INCLUDED
