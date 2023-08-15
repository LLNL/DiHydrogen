 ////////////////////////////////////////////////////////////////////////////////
 // Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
 // DiHydrogen Project Developers. See the top-level LICENSE file for details.
 //
 // SPDX-License-Identifier: Apache-2.0
 ////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

#include "h2/tensor/tensor_types.hpp"
#include "h2/meta/TypeList.hpp"

#ifdef HYDROGEN_HAVE_GPU
#include <cuda_runtime_api.h>
#endif

// List of device types that will be tested by Catch2.
// The raw CPU/GPUDev_t can be used in TEMPLATE_TEST_CASE and the
// AllDevList can be used in TEMPLATE_LIST_TEST_CASE.
using CPUDev_t = std::integral_constant<h2::Device, h2::Device::CPU>;
#ifdef HYDROGEN_HAVE_GPU
using GPUDev_t = std::integral_constant<h2::Device, h2::Device::GPU>;
#endif
using AllDevList = h2::meta::TypeList <CPUDev_t
#ifdef HYDROGEN_HAVE_GPU
                                   , GPUDev_t
#endif
                                    >;

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

#ifdef HYDROGEN_HAVE_GPU
// Note: These are not at all efficient, and meant only for testing
// with small test cases.
template <typename T>
struct Accessor<T, h2::Device::GPU>
{
  static T read_ele(const T* buf, std::size_t i)
  {
    T val;
    H_CHECK_CUDA(cudaMemcpy(&val, buf + i, sizeof(T), cudaMemcpyDeviceToHost));
    return val;
  }
  static void write_ele(T* buf, std::size_t i, const T& val)
  {
    H_CHECK_CUDA(cudaMemcpy(buf + i, &val, sizeof(T), cudaMemcpyHostToDevice));
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

template <typename T>
struct DeviceBuf<T, h2::Device::GPU>
{
  DeviceBuf(std::size_t size)
  {
    if (size > 0)
    {
      H_CHECK_CUDA(cudaMalloc(&buf, sizeof(T) * size));
    }
  }
  ~DeviceBuf()
  {
    if (buf)
    {
      H_CHECK_CUDA(cudaFree(buf));
    }
  }

  T* buf = nullptr;
};
