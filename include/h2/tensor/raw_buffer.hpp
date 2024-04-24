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

#include <algorithm>
#include <memory>
#include <new>
#include <ostream>
#include <unordered_map>
#include <cstddef>

#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/typename.hpp"
#include "h2/core/sync.hpp"
#include "h2/core/device.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif

namespace h2 {

namespace internal
{

// TODO: Use proper memory pools (probably Hydrogen's).

template <typename T, Device Dev>
struct Allocator {
  static T* allocate(std::size_t size, const ComputeStream& stream);
  static void deallocate(T* buf, const ComputeStream& stream);
};

template <typename T>
struct Allocator<T, Device::CPU> {
  static T* allocate(std::size_t size, const ComputeStream&) {
    return new T[size];
  }

  static void deallocate(T* buf, const ComputeStream&) {
    delete[] buf;
  }
};

#ifdef H2_HAS_GPU
template <typename T>
struct Allocator<T, Device::GPU>
{
  static T* allocate(std::size_t size, const ComputeStream& stream)
  {
    T* buf = nullptr;
    // FIXME: add H2_CHECK_GPU...
    H2_ASSERT(gpu::default_cub_allocator().DeviceAllocate(
                  reinterpret_cast<void**>(&buf),
                  size*sizeof(T),
                  stream.get_stream<Device::GPU>()) == 0,
              std::runtime_error,
              "CUB allocation failed.");
    return buf;
  }

  static void deallocate(T* buf, const ComputeStream&)
  {
    H2_ASSERT(gpu::default_cub_allocator().DeviceFree(buf) == 0,
              std::runtime_error,
              "CUB deallocation failed.");
  }
};
#endif

}  // namespace internal

/**
 * Manage a raw buffer of data on a device.
 */
template <typename T>
class RawBuffer {
public:

  RawBuffer(Device dev, const ComputeStream& stream_)
    : RawBuffer(dev, 0, false, stream_)
  {}

  RawBuffer(Device dev,
            std::size_t size,
            bool defer_alloc,
            const ComputeStream& stream_)
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
            const ComputeStream& stream_)
      : buffer(external_buffer),
        buffer_size(size),
        unowned_buffer(true),
        buffer_device(dev),
        stream(stream_)
  {}

  ~RawBuffer() { release(); }

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
      for (const auto& [other_stream, event] : pending_streams)
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
#else  // H2_HAS_GPU
    if (buffer) {
      if (!unowned_buffer) {
        internal::Allocator<T, Device::CPU>::deallocate(buffer, stream);
      }
      buffer = nullptr;
      unowned_buffer = false;
    }
#endif  // H2_HAS_GPU
  }

  Device get_device() const H2_NOEXCEPT { return buffer_device; }

  T* data() H2_NOEXCEPT { return buffer; }

  const T* data() const H2_NOEXCEPT { return buffer; }

  const T* const_data() const H2_NOEXCEPT { return buffer; }

  std::size_t size() const H2_NOEXCEPT { return buffer_size; }

  const ComputeStream& get_stream() const H2_NOEXCEPT { return stream; }

  void set_stream(const ComputeStream& stream_) { stream = stream_; }

  /**
   * Inform the RawBuffer that a stream is no longer using the
   * RawBuffer, but may have pending operations, and therefore needs to
   * be sync'd with the RawBuffer's stream.
   */
  void register_release([[maybe_unused]] const ComputeStream& other_stream)
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
        const auto& event = pending_streams[other_stream];
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
  T* buffer;  /**< Internal buffer. */
  std::size_t buffer_size;  /**< Number of elements in buffer. */
  bool unowned_buffer;      /**< Whether buffer is externally managed. */
  Device buffer_device;     /**< Device on which buffer was allocated. */
  ComputeStream stream;     /**< Device stream for synchronization. */

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
inline std::ostream& operator<<(std::ostream& os, const RawBuffer<T>& buf)
{
  os << "RawBuffer<" << TypeName<T>()
     << ", " << buf.get_device()
     << ", " << buf.get_stream()
     << ">(" << buf.const_data() << ", " << buf.size() << ")";
  return os;
}

namespace internal
{

template <typename T, Device Dev>
struct DeviceBufferPrinter
{
  DeviceBufferPrinter(const T* buf_, std::size_t size_) : buf(buf_), size(size_) {}

  void print(std::ostream& os)
  {
    os << "<" << Dev << " buffer of size " << size << ">";
  }

  const T* buf;
  std::size_t size;
};

template <typename T>
struct DeviceBufferPrinter<T, Device::CPU>
{
  DeviceBufferPrinter(const T* buf_, std::size_t size_) : buf(buf_), size(size_) {}

  void print(std::ostream& os)
  {
    for (std::size_t i = 0; i < size; ++i)
    {
      os << buf[i];
      if (i != size - 1)
      {
        os << ", ";
      }
    }
  }

  const T* buf;
  std::size_t size;
};

}  // namespace internal

/** Print the contents of a RawBuffer. */
template <typename T>
inline std::ostream& raw_buffer_contents(std::ostream& os,
                                         const RawBuffer<T>& buf)
{
  H2_DEVICE_DISPATCH_SAME(
    buf.get_device(),
    (internal::DeviceBufferPrinter<T, Dev>(buf.const_data(), buf.size()).print(os)));
  return os;
}

}  // namespace h2
