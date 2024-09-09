////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Routines for filling buffers or tensors with values.
 */

#include <h2_config.hpp>

#include "h2/core/sync.hpp"
#include "h2/core/types.hpp"
#include "h2/tensor/tensor.hpp"

#include <cstring>

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif

namespace h2
{

/**
 * Fill data with zeros.
 *
 * This is usable on any storage type: It just writes 0 bytes.
 *
 * If stream is a GPU stream, this will be asynchronous.
 */
template <typename T>
void zero(T* data, const ComputeStream& stream, std::size_t count)
{
  H2_ASSERT_DEBUG(count == 0 || data != nullptr, "Null buffers");
  static_assert(IsH2StorageType_v<T> || std::is_same_v<T*, void*>,
                "Attempt to zero a buffer with a non-storage type");

  H2_DEVICE_DISPATCH(
    stream.get_device(),
    std::memset(data, 0, std::is_same_v<T*, void*> ? count : count * sizeof(T)),
    gpu::mem_zero(data, count, stream.get_stream<Dev>()));
}

/**
 * Fill tensor with zeros.
 *
 * This is usable on any storage type: It just writes 0 bytes.
 *
 * If the tensor is on a GPU, this will be asynchronous.
 */
template <typename T>
void zero(Tensor<T>& tensor)
{
  if (tensor.is_contiguous())
  {
    zero(tensor.data(), tensor.get_stream(), tensor.numel());
  }
  else
  {
    throw H2FatalException("Zero not implemented for non-contiguous tensors");
  }
}

/**
 * Fill tensor with zeros.
 *
 * Unlike other `zero` versions, this is only usable for compute types.
 *
 * If the tensor is on a GPU, this will be asynchronous.
 */
void zero(BaseTensor& tensor);

namespace impl
{

template <typename T>
void fill_impl(CPUDev_t, Tensor<T>& tensor, const T& val);
#ifdef H2_HAS_GPU
template <typename T>
void fill_impl(GPUDev_t, Tensor<T>& tensor, const T& val);
#endif

} // namespace impl

/**
 * Fill tensor with a given value.
 *
 * This is usable only for compute types.
 *
 * If the tensor is on a GPU, this will be asynchronous. `val` does not
 * need to be on the GPU.
 */
template <typename T>
void fill(Tensor<T>& tensor, const T& val)
{
  H2_DEVICE_DISPATCH_SAME(tensor.get_device(),
                          impl::fill_impl(DeviceT_v<Dev>, tensor, val));
}

/**
 * Fill tensor with a given value.
 *
 * This is usable only for compute types.
 *
 * If the tensor is on a GPU, this will be asynchronous. `val` does not
 * need to be on the GPU.
 */
template <typename T>
void fill(BaseTensor& tensor, const T& val);

} // namespace h2
