////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Defines core device types and related information.
 */

#include <h2_config.hpp>

#include "h2/meta/TypeList.hpp"

#include <El.hpp>

#include <type_traits>

namespace hydrogen
{

/** Support printing Device. */
inline std::ostream& operator<<(std::ostream& os, const Device& dev)
{
  switch (dev)
  {
  case Device::CPU: os << "CPU"; break;
#ifdef H2_HAS_GPU
  case Device::GPU: os << "GPU"; break;
#endif
  default: os << "Unknown"; break;
  }
  return os;
}
} // namespace hydrogen

namespace h2
{

/**
 * Define the underlying compute device type.
 */
using Device = El::Device; // Leverage Hydrogen's device typing.

/**
 * Helper to support tagged dispatch based on the device.
 */
template <Device Dev>
struct DeviceTag
{
  static constexpr Device device = Dev;
  using device_t = std::integral_constant<Device, Dev>;
  static constexpr device_t device_t_v = {};
};

/** Helper to get an instance of CPUDev_t or GPUDev_t. */
template <Device Dev>
inline constexpr typename DeviceTag<Dev>::device_t DeviceT_v =
  DeviceTag<Dev>::device_t_v;

// Support representing devices as generic types.
using CPUDev_t = DeviceTag<Device::CPU>::device_t;
#ifdef H2_HAS_GPU
using GPUDev_t = DeviceTag<Device::GPU>::device_t;
#endif

#ifdef H2_HAS_GPU
// Type list of all device types.
using AllDevicesList = h2::meta::TL<CPUDev_t, GPUDev_t>;
#else
using AllDevicesList = h2::meta::TL<CPUDev_t>;
#endif

/**
 * The H2_DEVICE_DISPATCH macro is a helper for writing device dispatch
 * code that needs a template parameter.
 *
 * The first argument is code that, when evaluated, produces a runtime
 * device type (and is suitable for use in an `if` condition). The
 * second argument is the code block for CPU devices, and the third is
 * the code block for GPU devices.
 *
 * @warning When using these, beware of unprotected commas.
 */
#ifdef H2_HAS_GPU
#ifdef H2_DEBUG
#define H2_DEVICE_DISPATCH(device, cpu_code, gpu_code)                         \
  do                                                                           \
  {                                                                            \
    if ((device) == Device::CPU)                                               \
    {                                                                          \
      [[maybe_unused]] constexpr Device Dev = Device::CPU;                     \
      cpu_code;                                                                \
    }                                                                          \
    else if ((device) == Device::GPU)                                          \
    {                                                                          \
      [[maybe_unused]] constexpr Device Dev = Device::GPU;                     \
      gpu_code;                                                                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      throw H2Exception("Unknown device");                                     \
    }                                                                          \
  } while (0);
#else // H2_DEBUG
#define H2_DEVICE_DISPATCH(device, cpu_code, gpu_code)                         \
  do                                                                           \
  {                                                                            \
    if ((device) == Device::CPU)                                               \
    {                                                                          \
      [[maybe_unused]] constexpr Device Dev = Device::CPU;                     \
      cpu_code;                                                                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      [[maybe_unused]] constexpr Device Dev = Device::GPU;                     \
      gpu_code;                                                                \
    }                                                                          \
  } while (0);
#endif // H2_DEBUG
#else  // H2_HAS_GPU
#ifdef H2_DEBUG
#define H2_DEVICE_DISPATCH(device, cpu_code, gpu_code)                         \
  do                                                                           \
  {                                                                            \
    if ((device) == Device::CPU)                                               \
    {                                                                          \
      [[maybe_unused]] constexpr Device Dev = Device::CPU;                     \
      cpu_code;                                                                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      throw H2Exception("Unknown device");                                     \
    }                                                                          \
  } while (0);
#else // H2_DEBUG
#define H2_DEVICE_DISPATCH(device, cpu_code, gpu_code)                         \
  do                                                                           \
  {                                                                            \
    [[maybe_unused]] constexpr Device Dev = Device::CPU;                       \
    cpu_code;                                                                  \
  } while (0);
#endif // H2_DEBUG
#endif // H2_HAS_GPU

/**
 * Simplification of H2_DEVICE_DISPATCH when the code for both CPU and
 * GPU devices is the same.
 */
#define H2_DEVICE_DISPATCH_SAME(device, code)                                  \
  H2_DEVICE_DISPATCH(device, code, code)

/**
 * Similar to H2_DEVICE_DISPATCH, but a compile-time dispatch based on
 * a constexpr value.
 */
#ifdef H2_HAS_GPU
#define H2_DEVICE_DISPATCH_CONST(device, cpu_code, gpu_code)                   \
  do                                                                           \
  {                                                                            \
    if constexpr ((device) == Device::CPU)                                     \
    {                                                                          \
      cpu_code;                                                                \
    }                                                                          \
    else if constexpr ((device) == Device::GPU)                                \
    {                                                                          \
      gpu_code;                                                                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      throw H2Exception("Unknown device");                                     \
    }                                                                          \
  } while (0);
#else // H2_HAS_GPU
#define H2_DEVICE_DISPATCH_CONST(device, cpu_code, gpu_code)                   \
  do                                                                           \
  {                                                                            \
    if constexpr ((device) == Device::CPU)                                     \
    {                                                                          \
      cpu_code;                                                                \
    }                                                                          \
    else                                                                       \
    {                                                                          \
      throw H2Exception("Unknown device");                                     \
    }                                                                          \
  } while (0);
#endif // H2_HAS_GPU

} // namespace h2
