////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Manages a raw memory buffer.
 */

#include "h2/core/allocator.hpp"
#include "h2/core/device.hpp"
#include "h2/core/sync.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/typename.hpp"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <ostream>
#include <unordered_map>

#ifdef H2_HAS_GPU
#include "h2/gpu/runtime.hpp"
#endif

namespace h2
{

/**
 * Manage a raw buffer of data on a device.
 */
template <typename T>
class RawBuffer
{
public:
  RawBuffer(Device dev, ComputeStream const& stream_)
    : RawBuffer(dev, 0, false, stream_)
  {}

  RawBuffer(Device dev,
            std::size_t size,
            bool defer_alloc,
            ComputeStream const& stream_)
    : buffer(nullptr),
      buffer_size(size),
      unowned_buffer(false),
      buffer_device(dev),
      stream(stream_)
  {
    if (!defer_alloc)
    {
      ensure();
    }
  }

  RawBuffer(Device dev,
            T* external_buffer,
            std::size_t size,
            ComputeStream const& stream_)
    : buffer(external_buffer),
      buffer_size(size),
      unowned_buffer(true),
      buffer_device(dev),
      stream(stream_)
  {}

  ~RawBuffer() { H2_TERMINATE_ON_THROW_ALWAYS(release()); }

  /** Allocate memory if the buffer is not present. */
  void ensure()
  {
    if (buffer_size && !buffer && !unowned_buffer)
    {
      H2_DEVICE_DISPATCH_SAME(
        buffer_device,
        (buffer = internal::Allocator<T, Dev>::allocate(buffer_size, stream)));
    }
  }

  /**
   * Deallocate allocated memory.
   *
   * If the buffer is external, it will not be deallocated, but this
   * RawBuffer will no longer refer to it. Subsequent calls to `ensure`
   * will allocate a fresh buffer.
   */
  void release()
  {
#ifdef H2_HAS_GPU
    if (buffer)
    {
      // Wait for all pending operations.
      for (auto const& [other_stream, event] : pending_streams)
      {
        stream.wait_for(event);
      }

      if (!unowned_buffer)
      {
        H2_DEVICE_DISPATCH_SAME(
          buffer_device,
          (internal::Allocator<T, Dev>::deallocate(buffer, stream)));
      }
      buffer = nullptr;
      unowned_buffer = false;
    }
    // Clear all sync registrations.
    pending_streams.clear();
#else   // H2_HAS_GPU
    if (buffer)
    {
      if (!unowned_buffer)
      {
        internal::Allocator<T, Device::CPU>::deallocate(buffer, stream);
      }
      buffer = nullptr;
      unowned_buffer = false;
    }
#endif  // H2_HAS_GPU
  }

  Device get_device() const H2_NOEXCEPT { return buffer_device; }

  T* data() H2_NOEXCEPT { return buffer; }

  T const* data() const H2_NOEXCEPT { return buffer; }

  T const* const_data() const H2_NOEXCEPT { return buffer; }

  std::size_t size() const H2_NOEXCEPT { return buffer_size; }

  ComputeStream const& get_stream() const H2_NOEXCEPT { return stream; }

  void set_stream(ComputeStream const& stream_) { stream = stream_; }

  /**
   * Inform the RawBuffer that a stream is no longer using the
   * RawBuffer, but may have pending operations, and therefore needs to
   * be sync'd with the RawBuffer's stream.
   */
  void register_release([[maybe_unused]] ComputeStream const& other_stream)
  {
    // When we only have CPU devices, there is nothing to do.
#ifdef H2_HAS_GPU
    if (stream == other_stream)
    {
      // We are already ordered on our stream, so no need to manage it.
      return;
    }
    else if (other_stream.get_device() == Device::GPU)
    {
      // Only need to synchronize with GPU streams.
      // Check whether we already have a sync object for this stream.
      if (pending_streams.count(other_stream))
      {
        // Update the existing event.
        auto const& event = pending_streams[other_stream];
        other_stream.add_sync_point<Device::GPU, Device::GPU>(event);
      }
      else
      {
        // Create and record an event on the stream.
        SyncEventRAII event{Device::GPU};
        other_stream.add_sync_point<Device::GPU, Device::GPU>(event);
        pending_streams.emplace(other_stream, std::move(event));
      }
    }
#endif  // H2_HAS_GPU
  }

private:
  T* buffer;               /**< Internal buffer. */
  std::size_t buffer_size; /**< Number of elements in buffer. */
  bool unowned_buffer;     /**< Whether buffer is externally managed. */
  Device buffer_device;    /**< Device on which buffer was allocated. */
  ComputeStream stream;    /**< Device stream for synchronization. */

#ifdef H2_HAS_GPU
  /**
   * Record of streams which no longer reference this buffer, and an
   * event recorded on each stream at the point of deregistration.
   *
   * This is only needed with GPU support, as CPU streams are
   * inherently ordered.
   */
  std::unordered_map<ComputeStream, SyncEventRAII> pending_streams;
#endif
};

/** Support printing RawBuffer. */
template <typename T>
inline std::ostream& operator<<(std::ostream& os, RawBuffer<T> const& buf)
{
  os << "RawBuffer<" << TypeName<T>() << ", " << buf.get_device() << ", "
     << buf.get_stream() << ">(" << buf.const_data() << ", " << buf.size()
     << ")";
  return os;
}

namespace internal
{

template <typename T, Device Dev>
struct DeviceBufferPrinter
{
  static void print(T const* buf,
                    std::size_t size,
                    ComputeStream const& stream,
                    std::ostream& os);
};

template <typename T>
struct DeviceBufferPrinter<T, Device::CPU>
{
  static void
  print(T const* buf, std::size_t size, ComputeStream const&, std::ostream& os)
  {
    H2_ASSERT_DEBUG(size == 0 || buf != nullptr,
                    "Attempt to print null buffer");
    if (size > 0)
    {
      os << buf[0];
    }
    for (std::size_t i = 1; i < size; ++i)
    {
      os << ", " << buf[i];
    }
  }
};

#ifdef H2_HAS_GPU

template <typename T>
struct DeviceBufferPrinter<T, Device::GPU>
{
  static void print(T const* buf,
                    std::size_t size,
                    ComputeStream const& stream,
                    std::ostream& os)
  {
    H2_ASSERT_DEBUG(size == 0 || buf != nullptr,
                    "Attempt to print null buffer");
    H2_ASSERT_DEBUG(stream.get_device() == Device::GPU, "Not a GPU stream");
    if (size == 0)
    {
      return;
    }
    if (gpu::is_integrated())
    {
      stream.wait_for_this();
      // Regular CPU printer is fine.
      DeviceBufferPrinter<T, Device::CPU>::print(buf, size, stream, os);
    }
    else
    {
      internal::ManagedBuffer<T> cpu_buf(size, Device::CPU);
      gpu::mem_copy(
        cpu_buf.data(), buf, size, stream.get_stream<Device::GPU>());
      stream.wait_for_this();
      DeviceBufferPrinter<T, Device::CPU>::print(
        cpu_buf.data(), size, stream, os);
    }
  }
};

#endif  // H2_HAS_GPU

}  // namespace internal

/** Print the contents of a RawBuffer. */
template <typename T>
inline std::ostream& raw_buffer_contents(std::ostream& os,
                                         RawBuffer<T> const& buf)
{
  H2_DEVICE_DISPATCH_SAME(
    buf.get_device(),
    (internal::DeviceBufferPrinter<T, Dev>::print(
      buf.const_data(), buf.size(), buf.get_stream(), os)));
  return os;
}

}  // namespace h2
