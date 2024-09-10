////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Macro helpers for GPU code.
 */

#include <h2_config.hpp>

// Note: This file should be safe to include when not building with GPU
// support.

#if defined H2_HAS_GPU

#if (defined __CUDA_ARCH__ && __CUDA_ARCH__)                                   \
  || (defined __HIP_DEVICE_COMPILE__ && __HIP_DEVICE_COMPILE__)
#define H2_GPU_DEVICE_COMPILING 1
#else
#define H2_GPU_DEVICE_COMPILING 0
#endif

#endif  // H2_HAS_GPU

#if defined(__CUDACC__) || defined(__HIPCC__)

#define H2_GPU_HOST __host__
#define H2_GPU_DEVICE __device__
#define H2_GPU_HOST_DEVICE __host__ __device__
#define H2_GPU_GLOBAL __global__
#define H2_GPU_LAMBDA __host__ __device__
#define H2_GPU_FORCE_INLINE __forceinline__

#else

// These are defined to produce nothing when not building with a
// CUDA/HIP compiler, mostly to make clangd happy.
#define H2_GPU_HOST
#define H2_GPU_DEVICE
#define H2_GPU_HOST_DEVICE
#define H2_GPU_GLOBAL
#define H2_GPU_LAMBDA
#define H2_GPU_FORCE_INLINE inline

#endif  // defined(__CUDACC__) || defined(__HIPCC__)
