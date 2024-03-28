////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Routines to copy data and tensors.
 */


#include <h2_config.hpp>

#include <cstring>

#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor.hpp"
#include "h2/tensor/dist_tensor.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/runtime.hpp"
#include "h2/gpu/memory_utils.hpp"
#endif

namespace h2
{

// Low-level copy routines operating on buffers (raw pointers):

/**
 * Copy count elements on Dev from src to dst.
 *
 * If GPU buffers are involved, this will be asynchronous.
 */
template <Device DstDev, Device SrcDev, typename T>
void CopyBuffer(T* dst,
                const SyncInfo<DstDev>& dst_sync,
                const T* src,
                const SyncInfo<SrcDev>& src_sync,
                std::size_t count)
{
  if constexpr (SrcDev == Device::CPU && DstDev == Device::CPU)
  {
    std::memcpy(dst, src, count * sizeof(T));
  }
#ifdef H2_HAS_GPU
  else if constexpr (SrcDev == Device::GPU && DstDev == Device::GPU)
  {
    auto sync = El::MakeMultiSync(dst_sync, src_sync);
    gpu::mem_copy<T>(dst, src, count, dst_sync.Stream());
  }
  else if constexpr (SrcDev == Device::CPU && DstDev == Device::GPU)
  {
    // No sync needed in this case: The CPU is always synchronized and
    // the copy will be enqueued on the destination GPU stream.
    gpu::mem_copy<T>(dst, src, count, dst_sync.Stream());
  }
  else if constexpr (SrcDev == Device::GPU && DstDev == Device::CPU)
  {
    // No sync needed: Ditto.
    gpu::mem_copy<T>(dst, src, count, src_sync.Stream());
  }
#endif
  else
  {
    throw H2Exception("Unknown device combination");
  }
}

/** Special case where both buffers are on the same device. */
template <Device Dev, typename T>
void CopyBuffer(T* dst,
                const SyncInfo<Dev>& dst_sync,
                const T* src,
                const SyncInfo<Dev>& src_sync,
                std::size_t count)
{
  CopyBuffer<Dev, Dev>(src, src_sync, dst, dst_sync, count);
}

// General copy routines:


// ToDevice routines:



}
