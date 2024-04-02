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
#include <ostream>
#include <unordered_map>
#include <cstddef>

#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/typename.hpp"
#include "h2/core/sync.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/sync.hpp"
#endif

namespace h2 {

namespace internal
{

// TODO: Use proper memory pools (probably Hydrogen's).

template <typename T, Device Dev>
struct Allocator {
  static T* allocate(std::size_t size, const ComputeStream<Dev>& stream);
  static void deallocate(T* buf, const ComputeStream<Dev>& stream);
};

template <typename T>
struct Allocator<T, Device::CPU> {
  static T* allocate(std::size_t size, const ComputeStream<Device::CPU>&) {
    return new T[size];
  }

  static void deallocate(T* buf, const ComputeStream<Device::CPU>&) {
    delete[] buf;
  }
};

#ifdef H2_HAS_GPU
template <typename T>
struct Allocator<T, Device::GPU>
{
  static T* allocate(std::size_t size, const ComputeStream<Device::GPU>& stream)
  {
    T* buf = nullptr;
    // FIXME: add H2_CHECK_GPU...
    H2_ASSERT(gpu::default_cub_allocator().DeviceAllocate(
                  reinterpret_cast<void**>(&buf),
                  size*sizeof(T),
                  stream.get_stream()) == 0,
              std::runtime_error,
              "CUB allocation failed.");
    return buf;
  }

  static void deallocate(T* buf, const ComputeStream<Device::GPU>&)
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
template <typename T, Device Dev>
class RawBuffer {
public:
  RawBuffer(const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : buffer(nullptr), buffer_size(0), stream(stream_), unowned_buffer(false)
  {}
  RawBuffer(std::size_t size, bool defer_alloc = false,
            const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : buffer(nullptr), buffer_size(size), stream(stream_), unowned_buffer(false)
  {
    if (!defer_alloc)
    {
      ensure();
    }
  }
  RawBuffer(T* external_buffer, std::size_t size,
            const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : buffer(external_buffer), buffer_size(size), stream(stream_),
      unowned_buffer(true)
  {}

  ~RawBuffer() { release(); }

  /** Allocate memory if the buffer is not present. */
  void ensure()
  {
    if (buffer_size && !buffer && !unowned_buffer)
    {
      buffer = internal::Allocator<T, Dev>::allocate(buffer_size, stream);
    }
  }

  /**
   * Deallocate allocated memory.
   *
   * If the buffer is external, it will not be deallocated, but this
   * RawBuffer will no longer refer to it. Subsequent calls to `ensure`
   * will allocate a fresh buffer.
   */
  void release() {
    if (buffer)
    {
#ifdef H2_HAS_GPU
      if constexpr (Dev == Device::GPU)
      {
        // Have sync_info wait on all other syncs.
        for (const auto& [other_stream, event] : pending_streams)
        {
          stream.wait_for(event);
        }
      }
#endif
      if (!unowned_buffer)
      {
        internal::Allocator<T, Dev>::deallocate(buffer, stream);
      }
      buffer = nullptr;
      unowned_buffer = false;
    }
#ifdef H2_HAS_GPU
    if constexpr (Dev == Device::GPU)
    {
      // Clear all recorded syncs.
      pending_streams.clear();
    }
#endif
  }

  T* data() H2_NOEXCEPT { return buffer; }

  const T* data() const H2_NOEXCEPT { return buffer; }

  const T* const_data() const H2_NOEXCEPT { return buffer; }

  std::size_t size() const H2_NOEXCEPT { return buffer_size; }

  ComputeStream<Dev> get_stream() const H2_NOEXCEPT { return stream; }

  void set_stream(const ComputeStream<Dev>& stream_) { stream = stream_; }

  /**
   * Inform the RawBuffer that a stream is no longer using the
   * RawBuffer, but may have pending operations, and therefore needs to
   * be sync'd with the RawBuffer's stream.
   */
  void register_release(const ComputeStream<Dev>& other_stream)
  {
    // For CPU devices, we don't need to do anything.
#ifdef H2_HAS_GPU
    if constexpr (Dev == Device::GPU)
    {
      // We are already ordered on our stream, so no need to manage it.
      if (other_stream == stream)
      {
        return;
      }
      // Check whether we already have saved a sync object with the
      // same stream.
      if (pending_streams.count(other_stream))
      {
        // Update the event to capture any new work.
        other_stream.add_sync_point(pending_streams[other_stream]);
      }
      else
      {
        // Create and record an event on the stream.
        SyncEventRAII<Dev> event;
        other_stream.add_sync_point(event);
        pending_streams.emplace(other_stream, std::move(event));
      }
    }
#endif
  }

private:
  T* buffer;  /**< Internal buffer. */
  std::size_t buffer_size;  /**< Number of elements in buffer. */
  ComputeStream<Dev> stream;  /**< Synchronization management. */
  bool unowned_buffer;  /**< Whether buffer is externally managed. */
#ifdef H2_HAS_GPU
  /**
   * Streams and a recorded event that no longer reference this buffer.
   */
  std::unordered_map<ComputeStream<Dev>, SyncEventRAII<Dev>> pending_streams;
#endif
};

/** Support printing RawBuffer. */
template <typename T, Device Dev>
inline std::ostream& operator<<(std::ostream& os, const RawBuffer<T, Dev>& buf)
{
  os << "RawBuffer<" << TypeName<T>() << ", " << Dev << ">(" << buf.data()
     << ", " << buf.size() << ")";
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
template <typename T, Device Dev>
inline std::ostream& raw_buffer_contents(std::ostream& os,
                                         const RawBuffer<T, Dev>& buf)
{
  internal::DeviceBufferPrinter<T, Dev>(buf.const_data(), buf.size()).print(os);
  return os;
}

}  // namespace h2
