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
void dispatch_call(const DispatchFunctionEntry& func, Args&&... args)
{
  void* func_args[] = {(void*) &args...};
  func.caller(func.func_ptr, func_args);
}

// Note: "Native" dispatch keys are only for use with native H2 compute
// types. Regular dispatch keys work with anything with a token from
// h2::TypeInfo (including native compute types).
// A native dispatch key may *not* be converted to a regular one, or
// vice-versa.

/**
 * Number of bits needed to uniquely represent all native compute types.
 */
constexpr std::size_t dispatch_bits_per_native_compute_type =
    ceillog2(NumComputeTypes);
/**
 * Number of bits needed to uniquely represent all compute types.
 */
constexpr std::size_t dispatch_bits_per_compute_type =
    ceillog2(static_cast<std::size_t>(TypeInfo::max_token - 1));

/** Helper to construct a dispatch key with sanity checking. */
template <unsigned int num_types, std::size_t bits_per_compute_type>
struct DispatchKeyT_impl
{
  /** Number of bits needed to represent the dispatch key. */
  static constexpr unsigned int num_bits =
      bits_per_compute_type * num_types;
  /** Number of bytes needed to represent the dispatch key. */
  static constexpr unsigned int num_bytes = byteceil(num_bits);
  static_assert(
      num_bytes <= 8,
      "Cannot create a dispatch key that would require more than 8 bytes");
  /** Type to use for the dispatch key. */
  using type = typename UTypeForBytes<num_bytes>::type;
};

/**
 * Type for a native dispatch key that dispatches over num_types types.
 *
 * @warning Native dispatch keys are not comparable across different
 * numbers of types. (This is because you cannot distinguish a (float)
 * key from a (float, float) key: They are both 0.)
 */
template <unsigned int num_types>
using NativeDispatchKeyT = typename DispatchKeyT_impl<
    num_types,
    dispatch_bits_per_native_compute_type>::type;
/** Type for the largest native dispach key supported. */
using MaxNativeDispatchKeyT = typename UTypeForBytes<8>::type;
/** Maximum number of native dispatch types supported. */
constexpr std::size_t max_native_dispatch_types =
    (sizeof(MaxNativeDispatchKeyT) * 8) / dispatch_bits_per_native_compute_type;

/**
 * Type for a dispatch key that dispatches over native and non-native
 * compute types.
 */
using DispatchKeyT = typename UTypeForBytes<8>::type;
/** Maximum number of dispatch types supported. */
constexpr std::size_t max_dispatch_types =
    ((sizeof(DispatchKeyT) - 1) * 8) / dispatch_bits_per_compute_type;
/** Number of bits to shift to reach the top byte of DispatchKeyT. */
constexpr std::size_t dispatch_key_top_byte_shift =
    (sizeof(DispatchKeyT) - 1) * 8;

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
 * True if all arguments have a runtime type that is a native H2
 * compute type.
 */
template <typename... TypeInfoHavers>
bool all_h2_compute_types(const TypeInfoHavers&... args)
{
  return (is_h2_compute_type(get_type_info(args)) && ...);
}

/**
 * True if all arguments have a runtime type that is a compute type
 * (whether or not it is a native H2 compute type).
 */
template <typename... TypeInfoHavers>
bool all_compute_types(const TypeInfoHavers&... args)
{
  return (is_compute_type(get_type_info(args)) && ...);
}

/**
 * True if a non-native H2 compute type is present.
 */
template <typename... TypeInfoHavers>
bool contains_nonnative_compute_type(const TypeInfoHavers&... args)
{
  H2_ASSERT_DEBUG(all_compute_types(std::forward<TypeInfoHavers>(args)...),
                  "Cannot check for non-native compute types when non-compute "
                  "types are present");
  return ((is_compute_type(get_type_info(args)) && !is_h2_compute_type(args))
          && ...);
}

/** Construct a native dispatch key for dispatching on tokens. */
template <std::size_t N>
constexpr NativeDispatchKeyT<N>
get_native_dispatch_key(const std::array<TypeInfo::TokenType, N>& tokens)
{
  NativeDispatchKeyT<N> dispatch_key = 0;
  // Shift tokens, with the first being leftmost, to construct the key.
  for (std::size_t i = 0; i < N; ++i)
  {
    dispatch_key |= tokens[i]
                    << (dispatch_bits_per_native_compute_type * (N - 1 - i));
  }
  return dispatch_key;
}

/** Construct a native dispatch key for dispatching on args. */
template <typename... TypeInfoHavers>
constexpr NativeDispatchKeyT<sizeof...(TypeInfoHavers)>
get_native_dispatch_key(const TypeInfoHavers&... args)
{
  H2_ASSERT_DEBUG(
      all_h2_compute_types(args...),
      "Cannot construct native dispatch keys for non-native compute types");
  std::array<TypeInfo::TokenType, sizeof...(TypeInfoHavers)> tokens = {
      {get_type_token(args)...}};
  return get_native_dispatch_key(tokens);
}

/** Add a dispatch entry to the dispatch table for the name and key. */
void add_dispatch_entry(const std::string& name,
                        const DispatchKeyT& dispatch_key,
                        const DispatchFunctionEntry& dispatch_entry);

/** Return true if a dispatch entry exists for the name and key. */
bool has_dispatch_entry(const std::string& name,
                        const DispatchKeyT& dispatch_key);

/**
 * Return the dispatch entry for name and key.
 *
 * Throws if the entry is not present.
 */
const DispatchFunctionEntry&
get_dispatch_entry(const std::string& name,
                   const DispatchKeyT& dispatch_key);

/**
 * Call the dispatch entry for name and key with the given arguments.
 *
 * Throws if the entry is not present.
 */
template <typename... Args>
void call_dispatch_entry(const std::string& name,
                         const DispatchKeyT& dispatch_key,
                         Args&&... args)
{
  auto entry = get_dispatch_entry(name, dispatch_key);
  dispatch_call(entry, std::forward<Args>(args)...);
}

/** Construct a dispatch key for dispatching on tokens. */
template <std::size_t N>
constexpr DispatchKeyT
get_dispatch_key(const std::array<TypeInfo::TokenType, N>& tokens)
{
  static_assert(N <= max_dispatch_types,
                "Attempt to get dispatch key for too many types");
  DispatchKeyT dispatch_key = N << dispatch_key_top_byte_shift;
  // Shift tokens, with the first being leftmost, to construct the key.
  for (std::size_t i = 0; i < N; ++i)
  {
    dispatch_key |= tokens[i]
                    << (dispatch_bits_per_compute_type * (N - 1 - i));
  }
  return dispatch_key;
}

}  // namespace internal

/** Construct a dispatch key for dispatching on args. */
template <typename... TypeInfoHavers>
constexpr internal::DispatchKeyT
get_dispatch_key(const TypeInfoHavers&... args)
{
  H2_ASSERT_DEBUG(internal::all_compute_types(args...),
                  "Cannot construct dispatch keys for non-compute types");
  static_assert(sizeof...(args) <= internal::max_dispatch_types,
                "Attempt to get dispatch key for too many types");
  std::array<TypeInfo::TokenType, sizeof...(TypeInfoHavers)> tokens = {
      {internal::get_type_token(args)...}};
  return internal::get_dispatch_key(tokens);
}

/**
 * Register a function for dynamic dispatch.
 *
 * @param name[in] Name for the dispatch registry.
 * @param dispatch_key[in] Dispatch key representing the types this
 * function should be dispatched for.
 * @param func[in] Pointer to the function to dispatch to.
 */
template <typename... Args>
void dispatch_register(const std::string& name,
                       const internal::DispatchKeyT& dispatch_key,
                       void (*func)(Args...))
{
  internal::DispatchFunctionEntry dispatch_entry{
      reinterpret_cast<void*>(func),
      &internal::DispatchFunctionWrapper<void, Args...>::call};
  add_dispatch_entry(name, dispatch_key, dispatch_entry);
}

/** Unregister a dynamic dispatch entry. */
void dispatch_unregister(const std::string& name,
                         const internal::DispatchKeyT& dispatch_key);

/** Wrapper for dispatching on a set number of types. */
template <std::size_t num_types>
struct DispatchOn
{
  template <typename... Args>
  DispatchOn(const Args&... args)
      : tokens{{internal::get_type_token(args)...}},
        all_native(internal::all_h2_compute_types(args...))
  {
    static_assert(
        sizeof...(args) == num_types,
        "Provided different number of types to dispatch on than expected");
    if (!internal::all_compute_types(args...))
    {
      throw H2FatalException("Attempt to dispatch on a non-compute type");
    }
  }

  std::array<TypeInfo::TokenType, num_types> tokens;
  bool all_native;
};

/**
 * Dispatch on dispatch_types and invoke the function with args.
 *
 * This will handle both native compute type and registered dispatch.
 */
template <std::size_t N, std::size_t num_types, typename... Args>
void do_dispatch(
    const std::array<internal::DispatchFunctionEntry, N>& dispatch_table,
    const std::string& name,
    const DispatchOn<num_types>& dispatch_types,
    Args&&... args)
{
  if (dispatch_types.all_native)
  {
    const auto native_dispatch_key =
        internal::get_native_dispatch_key(dispatch_types.tokens);
    H2_ASSERT_DEBUG(native_dispatch_key < dispatch_table.size(),
                    "Native dispatch key exceeds dispatch table size");
    internal::dispatch_call(dispatch_table[native_dispatch_key],
                            std::forward<Args>(args)...);
  }
  else
  {
    const auto dispatch_key = internal::get_dispatch_key(dispatch_types.tokens);
    internal::call_dispatch_entry(
        name, dispatch_key, std::forward<Args>(args)...);
  }
}

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
