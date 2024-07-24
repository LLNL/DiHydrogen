////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Various raw memory allocators and related utilities.
 */

#include <h2_config.hpp>

#include <new>
#include <optional>
#include <cstddef>

#include "h2/core/device.hpp"
#include "h2/core/sync.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif


namespace h2
{

namespace internal
{

// TODO: Use proper memory pools (probably Hydrogen's).

template <typename T, Device Dev>
struct Allocator
{
  static T* allocate(std::size_t size, const ComputeStream& stream);
  static void deallocate(T* buf, const ComputeStream& stream);
};

template <typename T>
struct Allocator<T, Device::CPU>
{
  static T* allocate(std::size_t size, const ComputeStream&)
  {
    return new T[size];
  }

  static void deallocate(T* buf, const ComputeStream&)
  {
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

/**
 * Helper class to wrap an allocation in RAII semantics.
 */
template <typename T, Device Dev>
class ManagedBuffer
{
public:
  ManagedBuffer(std::size_t size_,
                const std::optional<ComputeStream> stream_ = std::nullopt)
    : buf(nullptr), buf_size(size_), stream(stream_.value_or(ComputeStream{Dev}))
  {
    if (buf_size)
    {
      buf = Allocator<T, Dev>::allocate(buf_size, stream);
    }
  }

  ~ManagedBuffer()
  {
    if (buf)
    {
      Allocator<T, Dev>::deallocate(buf, stream);
    }
  }

  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ManagedBuffer(ManagedBuffer&&) = default;
  ManagedBuffer& operator=(ManagedBuffer&&) = default;

  T* data() H2_NOEXCEPT { return buf; }

  const T* data() const H2_NOEXCEPT { return buf; }

  const T* const_data() const H2_NOEXCEPT { return buf; }

  std::size_t size() const H2_NOEXCEPT { return size; }

  const ComputeStream& get_stream() const H2_NOEXCEPT { return stream; }

private:
  T* buf;
  std::size_t buf_size;
  ComputeStream stream;
};

}  // namespace internal

}  // namespace h2
