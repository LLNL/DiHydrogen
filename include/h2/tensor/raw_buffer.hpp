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
#include <vector>
#include <cstddef>

#include "h2/tensor/tensor_types.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif

namespace h2 {

namespace internal
{

// TODO: Use proper memory pools (probably Hydrogen's).

template <typename T, Device Dev>
struct Allocator {
  static T* allocate(std::size_t size, const SyncInfo<Dev>& sync);
  static void deallocate(T* buf, const SyncInfo<Dev>& sync);
};

template <typename T>
struct Allocator<T, Device::CPU> {
  static T* allocate(std::size_t size, const SyncInfo<Device::CPU>&) {
    return new T[size];
  }

  static void deallocate(T* buf, const SyncInfo<Device::CPU>&) {
    delete[] buf;
  }
};

#ifdef H2_HAS_GPU
template <typename T>
struct Allocator<T, Device::GPU>
{
  static T* allocate(std::size_t size, const SyncInfo<Device::GPU>& si)
  {
    T* buf = nullptr;
    // FIXME: add H2_CHECK_GPU...
    H2_ASSERT(gpu::default_cub_allocator().DeviceAllocate(
                  reinterpret_cast<void**>(&buf),
                  size*sizeof(T),
                  si.Stream()) == 0,
              std::runtime_error,
              "CUB allocation failed.");
    return buf;
  }

  static void deallocate(T* buf, const SyncInfo<Device::GPU>&)
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
  RawBuffer(const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : buffer(nullptr), buffer_size(0), sync_info(sync), unowned_buffer(false)
  {}
  RawBuffer(std::size_t size, bool defer_alloc = false,
            const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : buffer(nullptr), buffer_size(size), sync_info(sync), unowned_buffer(false)
  {
    if (!defer_alloc)
    {
      ensure();
    }
  }
  RawBuffer(T* external_buffer, std::size_t size,
            const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : buffer(external_buffer), buffer_size(size), sync_info(sync),
      unowned_buffer(true)
  {}

  ~RawBuffer() { release(); }

  /** Allocate memory if the buffer is not present. */
  void ensure()
  {
    if (buffer_size && !buffer && !unowned_buffer)
    {
      buffer = internal::Allocator<T, Dev>::allocate(buffer_size, sync_info);
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
        for (const auto& s : pending_syncs)
        {
          El::AddSynchronizationPoint(s, sync_info);
        }
      }
#endif
      if (!unowned_buffer)
      {
        internal::Allocator<T, Dev>::deallocate(buffer, sync_info);
      }
      buffer = nullptr;
      unowned_buffer = false;
    }
#ifdef H2_HAS_GPU
    if constexpr (Dev == Device::GPU)
    {
      // Clear all recorded syncs.
      pending_syncs.clear();
    }
#endif
  }

  T* data() H2_NOEXCEPT { return buffer; }

  const T* data() const H2_NOEXCEPT { return buffer; }

  const T* const_data() const H2_NOEXCEPT { return buffer; }

  std::size_t size() const H2_NOEXCEPT { return buffer_size; }

  SyncInfo<Dev> get_sync_info() const H2_NOEXCEPT { return sync_info; }

  void set_sync_info(const SyncInfo<Dev>& sync) { sync_info = sync; }

  /**
   * Inform the RawBuffer that sync is no longer using the RawBuffer,
   * but may have pending operations, and therefore need to be sync'd
   * with the RawBuffer's SyncInfo.
   */
  void register_release(const SyncInfo<Dev>& sync)
  {
    // For CPU devices, we don't need to do anything.
#ifdef H2_HAS_GPU
    if constexpr (Dev == Device::GPU)
    {
      // Check whether we already have saved a sync object with the
      // same stream.
      // Note: We use a vector because we do not expect there to be
      // many distinct sync objects here. (And we don't have to deal
      // with hash functions.)
      auto i = std::find_if(
          pending_syncs.begin(),
          pending_syncs.end(),
          [&](const SyncInfo<Dev>& s) { return s.Stream() == sync.Stream(); });
      if (i == pending_syncs.end())
      {
        pending_syncs.push_back(sync);
      }
      else
      {
        // Update the event to capture any new work.
        El::AddSynchronizationPoint(*i);
      }
    }
#endif
  }

private:
  T* buffer;  /**< Internal buffer. */
  std::size_t buffer_size;  /**< Number of elements in buffer. */
  SyncInfo<Dev> sync_info;  /**< Synchronization management. */
  bool unowned_buffer;  /**< Whether buffer is externally managed. */
#ifdef H2_HAS_GPU
  /** List of sync objects that no longer reference this buffer. */
  std::vector<SyncInfo<Dev>> pending_syncs;
#endif
};

/** Support printing RawBuffer. */
template <typename T, Device Dev>
inline std::ostream& operator<<(std::ostream& os, const RawBuffer<T, Dev>& buf)
{
  // TODO: Print the type along with the device.
  os << "RawBuffer<" << Dev << ">(" << buf.data() << ", " << buf.size() << ")";
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
