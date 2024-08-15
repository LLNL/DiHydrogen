////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Routines for internal H2 kernel dispatch.
 */

#include <h2_config.hpp>

#include <type_traits>

#include "h2/core/types.hpp"
#include "h2/core/device.hpp"


/**
 * Overview of H2 dispatch:
 *
 * Dispatch is the process whereby calls to a generic function/method
 * (one with, e.g., a `template <typename T>` parameter) are translated
 * to a call to a concrete instance of an underlying function. This is
 * done to avoid having our header files instantiate compute kernels
 * for any type (which would make compilation hard and optimization
 * challenging).
 *
 * H2 generally provides kernels for all its compute types (anything
 * for which `h2::IsComputeTupe_v` is true), however, users of the
 * library may extend this to support custom types for particular
 * functions without modifying H2 itself.
 *
 * There are two dispatch mechanisms depending on whether the type(s)
 * being dispatched on are known at compile-time or not.
 *
 * Static (compile-time) dispatch:
 *
 * If the type(s) are known at compile-time, dispatch is simple and is
 * mainly about following a particular style. In the H2 header
 * implementation of the templated API, simply call the underlying
 * implementation kernel. This should be declared (but not defined!) in
 * the same header in an `impl` namespace. Then, in the associated
 * source file, either provide a generic implementation followed by
 * explicit instantiations with compute types, or specializations of
 * the method for the function (or class/etc.). (These are not mutually
 * exclusive and you can do both.)
 *
 * An alternative is to define the generic method in the header as
 * deleted (`= delete`) and then provide specializaitons in the source
 * file. (This precludes a generic implementation, sadly.)
 *
 * If a user wishes to provide an implementation for a custom type, all
 * they need to do is provide their own specialization of the
 * implementation kernel (which will need to be explicitly scoped to
 * the `h2::impl` namespace). If the API is called with an unsupported
 * type, the user application will fail to link. (If instead the
 * generic implementation is `delete`d, it will be a compile error.)
 *
 * This all works because the top-level API is header-only, so the
 * compiler doesn't actually generate the call to the underlying
 * implementation until the user's application builds.
 *
 * The only overhead for this dispatch method is the cost of a function
 * call to the underlying implementation.
 *
 * Device-specific dispatch is typically handled via tag-dispatch on
 * the device type (`CPUDev_t`, etc.).
 *
 * See below for an example of this (used also in our unit testing).
 *
 * Dynamic (runtime) dispatch:
 *
 * Documentation TODO.
 */

#define H2_INSTANTIATE_DEV_1(device)            \
  PROTO(device, float);                         \
  PROTO(device, double);                        \
  PROTO(device, std::int32_t);                  \
  PROTO(device, std::uint32_t);
#define H2_INSTANTIATE_DEV_2(device)            \
  PROTO(device, float, float);                  \
  PROTO(device, float, double);                 \
  PROTO(device, float, std::int32_t);           \
  PROTO(device, float, std::uint32_t);          \
  PROTO(device, double, float);                 \
  PROTO(device, double, double);                \
  PROTO(device, double, std::int32_t);          \
  PROTO(device, double, std::uint32_t);         \
  PROTO(device, std::int32_t, float);           \
  PROTO(device, std::int32_t, double);          \
  PROTO(device, std::int32_t, std::int32_t);    \
  PROTO(device, std::int32_t, std::uint32_t);   \
  PROTO(device, std::uint32_t, float);          \
  PROTO(device, std::uint32_t, double);         \
  PROTO(device, std::uint32_t, std::int32_t);   \
  PROTO(device, std::uint32_t, std::uint32_t);

#define H2_INSTANTIATE_CPU_1 H2_INSTANTIATE_DEV_1(CPUDev_t)
#define H2_INSTANTIATE_CPU_2 H2_INSTANTIATE_DEV_2(CPUDev_t)

#ifdef H2_HAS_GPU

#define H2_INSTANTIATE_GPU_1 H2_INSTANTIATE_DEV_1(GPUDev_t)
#define H2_INSTANTIATE_GPU_2 H2_INSTANTIATE_DEV_2(GPUDev_t)

#define H2_INSTANTIATE_1                        \
  H2_INSTANTIATE_CPU_1                          \
  H2_INSTANTIATE_GPU_1

#define H2_INSTANTIATE_2                        \
  H2_INSTANTIATE_CPU_2                          \
  H2_INSTANTIATE_GPU_2

#else  // H2_HAS_GPU

#define H2_INSTANTIATE_1 H2_INSTANTIATE_CPU_1

#define H2_INSTANTIATE_2 H2_INSTANTIATE_CPU_2

#endif  // H2_HAS_GPU

namespace h2
{

//

}  // namespace h2

// *****
// Example of static dispatching (this is also used in unit tests).

namespace h2
{

namespace impl
{

// Declare the underlying implementation, with CPU and GPU versions.
template <typename T>
void dispatch_test_impl(CPUDev_t, T*);

#ifdef H2_HAS_GPU
template <typename T>
void dispatch_test_impl(GPUDev_t, T*);
#endif

}  // namespace impl

// Define the H2 public API:
template <typename T>
void dispatch_test(Device dev, T* v)
{
  H2_DEVICE_DISPATCH_SAME(dev, impl::dispatch_test_impl(DeviceT_v<Dev>, v));
}

}  // namespace h2

// End static dispatch example.
// *****
