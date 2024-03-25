////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

#include "h2/tensor/tensor_types.hpp"
#include "h2/meta/TypeList.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/memory_utils.hpp"
#endif

// List of device types that will be tested by Catch2.
// The raw CPU/GPUDev_t can be used in TEMPLATE_TEST_CASE and the
// AllDevList can be used in TEMPLATE_LIST_TEST_CASE.
using CPUDev_t = std::integral_constant<h2::Device, h2::Device::CPU>;
#ifdef H2_TEST_WITH_GPU
using GPUDev_t = std::integral_constant<h2::Device, h2::Device::GPU>;
using AllDevList = h2::meta::TL<CPUDev_t, GPUDev_t>;
#else
using AllDevList = h2::meta::TL<CPUDev_t>;
#endif

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
  static T read_ele(const T* buf, std::size_t i);
  static void write_ele(T* buf, std::size_t i, const T& val);
};

template <typename T>
struct Accessor<T, h2::Device::CPU>
{
  static T read_ele(const T* buf, std::size_t i) { return buf[i]; }
  static void write_ele(T* buf, std::size_t i, const T& val) { buf[i] = val; }
};

#ifdef H2_HAS_GPU
// Note: These are not at all efficient, and meant only for testing
// with small test cases.
template <typename T>
struct Accessor<T, h2::Device::GPU>
{
  static T read_ele(const T* buf, std::size_t i)
  {
    T val;
    ::h2::gpu::mem_copy(&val, buf + i, 1);
    return val;
  }
  static void write_ele(T* buf, std::size_t i, const T& val)
  {
    ::h2::gpu::mem_copy(buf + i, &val, 1);
  }
};
#endif

}  // namespace internal

// Helper to read a value from a buffer on a device.
// Equivalent to buf[i].
template <h2::Device Dev, typename T>
inline T read_ele(const T* buf, std::size_t i = 0)
{
  return internal::Accessor<T, Dev>::read_ele(buf, i);
}

// Helper to write a value from a buffer on a device.
// Equivalent to buf[i] = val.
template <h2::Device Dev, typename T>
inline void write_ele(T* buf, std::size_t i, const T& val)
{
  internal::Accessor<T, Dev>::write_ele(buf, i, val);
}

// Simple class to manage a buffer on a device.
template <typename T, h2::Device Dev>
struct DeviceBuf {};

template <typename T>
struct DeviceBuf<T, h2::Device::CPU>
{
  DeviceBuf(std::size_t size)
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

  T* buf = nullptr;
};

#ifdef H2_HAS_GPU
template <typename T>
struct DeviceBuf<T, h2::Device::GPU>
{
  DeviceBuf(std::size_t size)
  {
    if (size > 0)
    {
      H2_ASSERT(
          ::h2::gpu::default_cub_allocator().DeviceAllocate(
              reinterpret_cast<void**>(&buf),
              size*sizeof(T),
              /*stream=*/0) == 0,
          std::runtime_error,
          "CUB allocation failed.");
    }
  }
  ~DeviceBuf()
  {
    if (buf)
      static_cast<void>(::h2::gpu::default_cub_allocator().DeviceFree(buf));
  }

  T* buf = nullptr;
};
#endif // H2_HAS_GPU
