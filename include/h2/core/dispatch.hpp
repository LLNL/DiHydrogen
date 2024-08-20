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
#include <utility>

#include "h2/core/types.hpp"
#include "h2/core/device.hpp"
#include "h2/utils/IntegerMath.hpp"


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

namespace internal
{

/**
 * An entry in a dynamic dispatch table.
 *
 * This holds a function pointer (which will be dispatched to) and a
 * function pointer to a "trampoline" caller which can reconstruct the
 * true types of the function from a void*[] argument list.
 */
struct DispatchFunctionEntry
{
  void* func_ptr;
  void (*caller)(void*, void**);
};

/**
 * Wrapper to facilitate calling a type-erased function pointer.
 *
 * This is intended for use with `DispatchFunctionEntry`, which should
 * hold the original function pointer and a function pointer to the
 * `call` method of this class with the correct arguments.
 */
template <typename Ret, typename... Args>
struct DispatchFunctionWrapper
{
  using FuncT = Ret (*)(Args...);
  /** Call f (a function pointer) with the given arguments. */
  static void call(void* f, void** args)
  {
    FuncT func = reinterpret_cast<FuncT>(f);
    call_impl(func, args, std::index_sequence_for<Args...>{});
  }

private:
  /** Helper to invoke func with args, expanding it appropriately. */
  template <std::size_t... I>
  static void call_impl(FuncT func, void** args, std::index_sequence<I...>)
  {
    // TODO:
    // It is technically undefined behavior to cast from a void* to any
    // type other than what the original type was. In the common case
    // where we have a function taking a BaseTensor and an
    // implementation taking a Tensor<T> (which is what the BaseTensor
    // really is), we should cast to BaseTensor first and then downcast
    // to Tensor<T> (possibly with dynamic_cast for extra safety).
    func(*reinterpret_cast<std::remove_reference_t<Args>*>(args[I])...);
  }
};

/** Call the function in the dispatch entry with args. */
template <typename... Args>
void dispatch_call(DispatchFunctionEntry& func, Args&&... args)
{
  void* func_args[] = {(void*) &args...};
  func.caller(func.func_ptr, func_args);
}


/** Number of bits needed to uniquely represent all compute types. */
constexpr std::size_t dispatch_bits_per_compute_type =
    ceillog2(NumComputeTypes);

/** Helper to construct a dispatch key with sanity checking. */
template <unsigned int num_types>
struct DispatchKeyT_impl
{
  /** Number of bits needed to represent the dispatch key. */
  static constexpr unsigned int num_bits =
      dispatch_bits_per_compute_type * num_types;
  /** Number of bytes needed to represent the dispatch key. */
  static constexpr unsigned int num_bytes = byteceil(num_bits);
  static_assert(
      num_bytes <= 8,
      "Cannot create a dispatch key that would require more than8 bytes");
  /** Type to use for the dispatch key. */
  using type = typename UTypeForBytes<num_bytes>::type;
};

/** Type for a dispatch key that dispatches over num_types types. */
template <unsigned int num_types>
using DispatchKeyT = typename DispatchKeyT_impl<num_types>::type;

// TODO: Handle case where it's not a compute type.
// TODO: Versions with TypeInfo(const?).

/**
 * Extract the `TypeInfo` from something.
 *
 * The "something" (x) must be either a `TypeInfo` already or something
 * that has a `get_type_info` method.
 */
template <typename TypeInfoHaver>
inline TypeInfo get_type_info(const TypeInfoHaver& x)
{
  return x.get_type_info();
}

template <>
inline TypeInfo get_type_info<TypeInfo>(const TypeInfo& tinfo)
{
  return tinfo;
}

/** True if all arguments have a runtime type that is a compute type. */
template <typename... TypeInfoHavers>
bool all_h2_compute_types(const TypeInfoHavers&... args)
{
  return (is_h2_type(get_type_info(args)) && ...);
}

/**
 * Get the type token for x, which must meet the same requirements as
 * in `get_type_info`.
 */
template <typename TypeInfoHaver>
inline TypeInfo::TokenType get_type_token(const TypeInfoHaver& x)
{
  return get_type_info(x).get_token();
}

/**
 * Return the dispatch key for dispatching over args.
 */
template <typename... TypeInfoHavers>
constexpr DispatchKeyT<sizeof...(TypeInfoHavers)>
get_dispatch_key(const TypeInfoHavers&... args)
{
  std::array<TypeInfo::TokenType, sizeof...(TypeInfoHavers)> tokens = {
      {get_type_token(args)...}};
  DispatchKeyT<sizeof...(TypeInfoHavers)> dispatch_key = 0;
  // Shift tokens, with the first being leftmost, to construct the key.
  for (std::size_t i = 0; i < sizeof...(args); ++i)
  {
    dispatch_key |= tokens[i] << (dispatch_bits_per_compute_type
                                  * (sizeof...(TypeInfoHavers) - 1 - i));
  }
  return dispatch_key;
}

}  // namespace internal

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
