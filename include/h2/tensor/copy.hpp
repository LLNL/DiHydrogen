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
#include <type_traits>
#include "strided_memory.hpp"

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
                const ComputeStream<DstDev>& dst_stream,
                const T* src,
                const ComputeStream<SrcDev>& src_stream,
                std::size_t count)
{
  H2_ASSERT_DEBUG(count == 0 || (dst != nullptr && src != nullptr),
                  "Null buffers");
  if constexpr (SrcDev == Device::CPU && DstDev == Device::CPU)
  {
    std::memcpy(dst, src, count * sizeof(T));
  }
#ifdef H2_HAS_GPU
  else if constexpr (SrcDev == Device::GPU && DstDev == Device::GPU)
  {
    auto stream = create_multi_sync(dst_stream, src_stream);
    gpu::mem_copy<T>(dst, src, count, dst_stream.get_stream());
  }
  else if constexpr (SrcDev == Device::CPU && DstDev == Device::GPU)
  {
    // No sync needed in this case: The CPU is always synchronized and
    // the copy will be enqueued on the destination GPU stream.
    gpu::mem_copy<T>(dst, src, count, dst_stream.get_stream());
  }
  else if constexpr (SrcDev == Device::GPU && DstDev == Device::CPU)
  {
    // No sync needed: Ditto.
    gpu::mem_copy<T>(dst, src, count, src_stream.get_stream());
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
                const ComputeStream<Dev>& dst_stream,
                const T* src,
                const ComputeStream<Dev>& src_stream,
                std::size_t count)
{
  CopyBuffer<Dev, Dev, T>(src, src_stream, dst, dst_stream, count);
}

// General copy routines:

namespace internal
{

template <Device DstDev, Device SrcDev, typename T>
void CopySameT(Tensor<T, DstDev>& dst, const Tensor<T, SrcDev>& src)
{
  dst.resize(src.shape(), src.dim_types(), src.strides());
  dst.ensure();
  if (src.is_contiguous())
  {
    CopyBuffer<DstDev, SrcDev, T>(dst.data(),
                                  dst.get_stream(),
                                  src.const_data(),
                                  src.get_stream(),
                                  src.numel());
  }
  else
  {
    // TODO: We may be able to optimize the non-contiguous case.
    // For now, we just copy the entire buffer.
    CopyBuffer<DstDev, SrcDev, T>(
        dst.data(),
        dst.get_stream(),
        src.const_data(),
        src.get_stream(),
        get_extent_from_strides(src.shape(), src.strides()));
  }
}

}  // namespace internal

/**
 * Copy the contents of tensor `src` to `dst`.
 *
 * `dst` will be resized and will have its dimension types changed to
 * match `src`. If `SrcT` and `DstT` differ, data will be converted, if
 * possible. This will preserve strides, i.e., if `src` is not
 * contiguous, then `dst` will be too.
 *
 * If GPU buffers are involved, this will be asynchronous.
 */
template <typename DstT, Device DstDev, typename SrcT, Device SrcDev>
void Copy(Tensor<DstT, DstDev>& dst, const Tensor<SrcT, SrcDev>& src)
{
  // Copying an empty tensor is permitted, but you cannot copy a lazy
  // tensor that has not been ensure'd.
  if (src.is_empty())
  {
    dst.empty();
    return;
  }
  H2_ASSERT_ALWAYS(src.const_data() != nullptr,
                   "Cannot copy a non-empty tensor with no data");
  if constexpr (std::is_same_v<SrcT, DstT>)
  {
    internal::CopySameT<DstDev, SrcDev, DstT>(dst, src);
  }
  else
  {
    throw H2Exception("Data type conversion in Copy not currently supported");
  }
}

/** Runtime dispatch on device type for tensors. */
template <typename DstT, typename SrcT>
void Copy(BaseTensor<DstT>& dst, const BaseTensor<SrcT>& src)
{
  if (src.get_device() == Device::CPU && dst.get_device() == Device::CPU)
  {
    Copy<DstT, Device::CPU, SrcT, Device::CPU>(
        static_cast<Tensor<DstT, Device::CPU>>(dst),
        static_cast<Tensor<SrcT, Device::CPU>>(src));
  }
#ifdef H2_HAS_GPU
  else if (src.get_device() == Device::GPU && dst.get_device() == Device::GPU)
  {
    Copy<DstT, Device::GPU, SrcT, Device::GPU>(
        static_cast<Tensor<DstT, Device::GPU>>(dst),
        static_cast<Tensor<SrcT, Device::GPU>>(src));
  }
  else if (src.get_device() == Device::CPU && dst.get_device() == Device::GPU)
  {
    Copy<DstT, Device::GPU, SrcT, Device::CPU>(
        static_cast<Tensor<DstT, Device::GPU>>(dst),
        static_cast<Tensor<SrcT, Device::CPU>>(src));
  }
  else if (src.get_device() == Device::GPU && dst.get_device() == Device::CPU)
  {
    Copy<DstT, Device::CPU, SrcT, Device::GPU>(
        static_cast<Tensor<DstT, Device::CPU>>(dst),
        static_cast<Tensor<SrcT, Device::GPU>>(src));
  }
#endif
  else
  {
    throw H2Exception("Unknown device combination");
  }
}

}
