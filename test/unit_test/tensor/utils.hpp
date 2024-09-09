////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/core/device.hpp"
#include "h2/core/types.hpp"
#include "h2/meta/TypeList.hpp"
#include "h2/tensor/tensor_types.hpp"

#include <type_traits>

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif

// The core device types can be used in TEMPLATE_TEST_CASE and the
// typelists can be used in TEMPLATE_LIST_TEST_CASE.
using h2::CPUDev_t;
#ifdef H2_TEST_WITH_GPU
using h2::GPUDev_t;
using AllDevList = h2::meta::TL<CPUDev_t, GPUDev_t>;
#else
using AllDevList = h2::meta::TL<CPUDev_t>;
#endif
using AllDevPairsList = h2::meta::tlist::CartProdTL<AllDevList, AllDevList>;

// All pairs of compute types:
using AllComputeTypePairsList =
  h2::meta::tlist::CartProdTL<h2::ComputeTypes, h2::ComputeTypes>;
// All pairs of devices and compute types:
using AllDevComputeTypePairsList =
  h2::meta::tlist::CartProdTL<AllDevList, h2::ComputeTypes>;
// All devices x pairs of types:
using AllDevComputeTypePairsPairsList =
  h2::meta::tlist::CartProdTL<AllDevList, AllComputeTypePairsList>;

// Standard datatype to be used when testing.
// Note: When used with integers, floats are exact for any integer with
// absolute value less than 2^24.
using DataType = float;

namespace internal
{

// Helpers for managing buffers on different devices.

template <typename T, h2::Device Dev>
struct Accessor
{
  static T
  read_ele(const T* buf, std::size_t i, const h2::ComputeStream& stream);
  static void write_ele(T* buf,
                        std::size_t i,
                        const T& val,
                        const h2::ComputeStream& stream);
};

template <typename T>
struct Accessor<T, h2::Device::CPU>
{
  static T read_ele(const T* buf, std::size_t i, const h2::ComputeStream&)
  {
    return buf[i];
  }
  static void
  write_ele(T* buf, std::size_t i, const T& val, const h2::ComputeStream&)
  {
    buf[i] = val;
  }
  static void write_ele_nosync(T* buf,
                               std::size_t i,
                               const T& val,
                               const h2::ComputeStream&)
  {
    buf[i] = val;
  }
};

#ifdef H2_HAS_GPU
// Note: These are not at all efficient, and meant only for testing
// with small test cases.
template <typename T>
struct Accessor<T, h2::Device::GPU>
{
  static T
  read_ele(const T* buf, std::size_t i, const h2::ComputeStream& stream)
  {
    T val;
    ::h2::gpu::mem_copy(&val, buf + i, 1, stream.get_stream<h2::Device::GPU>());
    stream.wait_for_this();
    return val;
  }
  static void write_ele(T* buf,
                        std::size_t i,
                        const T& val,
                        const h2::ComputeStream& stream)
  {
    ::h2::gpu::mem_copy(buf + i, &val, 1, stream.get_stream<h2::Device::GPU>());
    stream.wait_for_this();
  }
  static void write_ele_nosync(T* buf,
                               std::size_t i,
                               const T& val,
                               const h2::ComputeStream& stream)
  {
    ::h2::gpu::mem_copy(
      buf + i, &val, 1, stream.get_stream<::h2::Device::GPU>());
  }
};
#endif

} // namespace internal

// Helper to read a value from a buffer on a device.
// Equivalent to buf[i].
template <h2::Device Dev, typename T>
inline T read_ele(const T* buf, std::size_t i, const h2::ComputeStream& stream)
{
  return internal::Accessor<T, Dev>::read_ele(buf, i, stream);
}

// Equivalent to buf[0].
template <h2::Device Dev, typename T>
inline T read_ele(const T* buf, const h2::ComputeStream& stream)
{
  return read_ele<Dev>(buf, 0, stream);
}

// Helper to write a value from a buffer on a device.
// Equivalent to buf[i] = val.
template <h2::Device Dev, typename T>
inline void
write_ele(T* buf, std::size_t i, const T& val, const h2::ComputeStream& stream)
{
  internal::Accessor<T, Dev>::write_ele(buf, i, val, stream);
}

template <h2::Device Dev, typename T>
inline void write_ele_nosync(T* buf,
                             std::size_t i,
                             const T& val,
                             const h2::ComputeStream& stream)
{
  internal::Accessor<T, Dev>::write_ele_nosync(buf, i, val, stream);
}

// Simple class to manage a buffer on a device.
template <typename T, h2::Device Dev>
struct DeviceBuf
{};

template <typename T>
struct DeviceBuf<T, h2::Device::CPU>
{
  DeviceBuf(std::size_t size_) : size(size_)
  {
    if (size > 0)
    {
      buf = new T[size];
    }
  }
  ~DeviceBuf()
  {
    if (buf)
    {
      delete[] buf;
    }
  }

  void fill(const T val)
  {
    for (std::size_t i = 0; i < size; ++i)
    {
      buf[i] = val;
    }
  }

  std::size_t size;
  T* buf = nullptr;
};

#ifdef H2_HAS_GPU

template <typename T>
struct DeviceBuf<T, h2::Device::GPU>
{
  DeviceBuf(std::size_t size_) : size(size_)
  {
    if (size > 0)
    {
      H2_ASSERT(::h2::gpu::default_cub_allocator().DeviceAllocate(
                  reinterpret_cast<void**>(&buf),
                  size * sizeof(T),
                  /*stream=*/0)
                  == 0,
                std::runtime_error,
                "CUB allocation failed.");
    }
  }
  ~DeviceBuf()
  {
    if (buf)
      static_cast<void>(::h2::gpu::default_cub_allocator().DeviceFree(buf));
  }

  void fill(const T val)
  {
    for (std::size_t i = 0; i < size; ++i)
    {
      write_ele<h2::Device::GPU>(buf, i, val, 0);
    }
  }

  std::size_t size;
  T* buf = nullptr;
};

#endif // H2_HAS_GPU
