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

#include <memory>
#include <type_traits>

#include "h2/tensor/copy_buffer.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor.hpp"
#include "h2/tensor/dist_tensor.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/runtime.hpp"
#endif

namespace h2
{

namespace internal
{

template <typename T>
void copy_same_type(Tensor<T>& dst, const Tensor<T>& src)
{
  dst.resize(src.shape(), src.dim_types(), src.strides());
  dst.ensure();
  if (src.is_contiguous())
  {
    copy_buffer<T>(dst.data(),
                   dst.get_stream(),
                   src.const_data(),
                   src.get_stream(),
                   src.numel());
  }
  else
  {
    // TODO: We may be able to optimize the non-contiguous case.
    // For now, we just copy the entire buffer.
    copy_buffer<T>(
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
template <typename DstT, typename SrcT>
void copy(Tensor<DstT>& dst, const Tensor<SrcT>& src)
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
    internal::copy_same_type<DstT>(dst, src);
  }
  else
  {
    throw H2Exception("Data type conversion in Copy not currently supported");
  }
}

/**
 * Return a version of tensor src that is accessible from a device.
 *
 * This may return either a copy of the tensor or a view of the
 * original tensor.
 *
 * A view may be returned when the tensor is already on the requested
 * device; or if the system is a truly unified memory system (like an
 * APU) where `src`'s device and `dev` share the same physical memory.
 * In the latter case, the view will have a different device from the
 * original tensor.
 *
 * An optional stream may be provided to control the stream the
 * returned tensor will be on. If it is not specified, the stream used
 * will be as follows:
 * - If `src` is already on `dev`, `src`'s stream will be used.
 * - Otherwise, `dev`'s default stream will be used.
 */
template <typename T>
std::unique_ptr<Tensor<T>> make_accessible_on_device(
    Tensor<T>& src,
    Device dev,
    const std::optional<ComputeStream> stream = std::nullopt)
{
  if (src.get_device() == dev)
  {
    auto view = src.view();
    if (stream.has_value())
    {
      view->set_stream(stream.value());
    }
    return view;
  }

  ComputeStream real_stream = stream.value_or(ComputeStream{dev});
#ifdef H2_HAS_GPU
  if (gpu::is_integrated())
  {
    // Return a view with the device changed.
    return std::make_unique<Tensor<T>>(src, dev, real_stream);
  }
  else
  {
    // Return a copy.
    // Create a new tensor on `dev` that has the same size.
    auto dst = std::make_unique<Tensor<T>>(
        dev, src.shape(), src.dim_types(), StrictAlloc, real_stream);
    copy(*dst, src);
    return dst;
  }
#else  // H2_HAS_GPU
  // No GPU support, but dev differs from the tensor's device.
  // This should not happen.
  throw H2Exception("Unknown device ", dev);
#endif  // H2_HAS_GPU
}

}
