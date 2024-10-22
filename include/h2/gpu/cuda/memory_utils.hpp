////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 *  Thin wrappers around cudaMem{cpy,set} functions. These are here so
 *  they can be inlined if possible.
 */
#include "h2_config.hpp"

#include "h2/gpu/logger.hpp"
#include "h2/gpu/pool_allocator.hpp"
#include "h2/gpu/runtime.hpp"

#include <cuda_runtime.h>

namespace h2
{
namespace gpu
{

using RawCUBAllocType = h2::PooledDeviceAllocator;

struct MemInfo
{
  size_t free;
  size_t total;
};

inline MemInfo mem_info()
{
  MemInfo info;
  H2_CHECK_CUDA(cudaMemGetInfo(&info.free, &info.total));
  return info;
}

inline void mem_copy(void* dst, void const* src, size_t bytes)
{
  H2_GPU_TRACE("cudaMemcpy(dst={}, src={}, bytes={}, kind=cudaMemcpyDefault)",
               dst,
               src,
               bytes);
  H2_CHECK_CUDA(cudaMemcpy(dst, src, bytes, cudaMemcpyDefault));
}

inline void
mem_copy(void* dst, void const* src, size_t bytes, DeviceStream stream)
{
  H2_GPU_TRACE("cudaMemcpyAsync(dst={}, src={}, bytes={}, "
               "kind=cudaMemcpyDefault, stream={})",
               dst,
               src,
               bytes,
               (void*) stream);
  H2_CHECK_CUDA(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, stream));
}

inline void mem_zero(void* mem, size_t bytes)
{
  H2_GPU_TRACE("cudaMemset(mem={}, value=0x0, bytes={})", mem, bytes);
  H2_CHECK_CUDA(cudaMemset(mem, 0x0, bytes));
}

inline void mem_zero(void* mem, size_t bytes, DeviceStream stream)
{
  H2_GPU_TRACE("cudaMemsetAsync(mem={}, value=0x0, bytes={}, stream={})",
               mem,
               bytes,
               (void*) stream);
  H2_CHECK_CUDA(cudaMemsetAsync(mem, 0x0, bytes, stream));
}

}  // namespace gpu
}  // namespace h2
