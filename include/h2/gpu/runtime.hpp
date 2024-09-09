////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 *  The public runtime API that is exposed for user and library
 *  consumption. These are lightweight wrappers around HIP or CUDA
 *  runtime functions. The are accessible in the h2::gpu namespace.
 *
 *  typedef {cuda,hip}Stream_t DeviceStream;
 *  typedef {cuda,hip}Event_t DeviceEvent;
 *
 *  int num_gpus();
 *  int current_gpu();
 *  void set_gpu(int id);
 *
 *  void init_runtime();
 *  void finalize_runtime();
 *  bool runtime_is_initialized();
 *  bool runtime_is_finalized();
 *  bool is_integrated();
 *
 *  bool ok(DeviceError) noexcept;
 *
 *  DeviceStream make_stream();
 *  DeviceStream make_stream_nonblocking();
 *  void destroy(DeviceStream);
 *  void record_event(DeviceEvent, DeviceStream);
 *
 *  DeviceEvent make_event();
 *  DeviceEvent make_event_notiming();
 *  void destroy(DeviceEvent);
 *
 *  void sync();             // Device Sync
 *  void sync(DeviceEvent);  // Sync on event.
 *  void sync(DeviceStream); // Sync on stream.
 *
 *  void launch_kernel(...)
 *
 *  Constants that may be useful:
 *  unsigned int max_grid_x, max_grid_y, max_grid_z
 *  unsigned int max_block_x, max_block_y, max_block_z
 *  unsigned int max_threads_per_block
 *  unsigned int warp_size
 */

#include "h2_config.hpp"

#include "h2/gpu/logger.hpp"
#include "h2/meta/TypeList.hpp"

#include <type_traits>

// This adds the runtime-specific stuff.
#if H2_HAS_CUDA
#include "cuda/runtime.hpp"
#elif H2_HAS_ROCM
#include "rocm/runtime.hpp"
#endif

// This declares the rest of the general API.
namespace h2
{
namespace gpu
{

int num_gpus();
int current_gpu();
void set_gpu(int id);

void init_runtime();
void finalize_runtime();
bool runtime_is_initialized();
bool runtime_is_finalized();

/** True if the CPU and GPU are one integrated platform (like an APU) */
bool is_integrated();

DeviceStream make_stream();
DeviceStream make_stream_nonblocking();
void destroy(DeviceStream);

DeviceEvent make_event();
DeviceEvent make_event_notiming();
void destroy(DeviceEvent);
void record_event(DeviceEvent, DeviceStream);

void sync();                           // Device Sync
void sync(DeviceEvent);                // Sync on event.
void sync(DeviceStream);               // Sync on stream.
void sync(DeviceStream, DeviceEvent);  // Sync stream on event.

namespace internal
{

template <typename TL>
struct is_convertible_t;

template <typename T1, typename T2>
struct is_convertible_t<meta::TL<T1, T2>>
  : std::bool_constant<std::is_convertible_v<std::remove_reference_t<T1>,
                                             std::remove_reference_t<T2>>>
{};

template <typename TL>
using is_convertible = meta::Force<is_convertible_t<TL>>;

template <typename T, typename = void>
struct is_maybe_lambda_or_functor_t : std::false_type
{};

// Won't work if operator() is overloaded... :/
template <typename T>
struct is_maybe_lambda_or_functor_t<T, std::void_t<decltype(&T::operator())>>
  : std::true_type
{};

template <typename T>
constexpr static bool is_maybe_lambda_or_functor =
  is_maybe_lambda_or_functor_t<T>::value;

template <typename T,
          typename = std::enable_if_t<is_maybe_lambda_or_functor<T>>>
std::string convert_for_fmt(const T& v) noexcept
{
  return "<callable>";
}

template <typename T,
          typename = std::enable_if_t<!is_maybe_lambda_or_functor<T>>,
          typename = void>
const T& convert_for_fmt(const T& v) noexcept
{
  return v;
}

template <typename T>
void* convert_for_fmt(T* const v) noexcept
{
  return reinterpret_cast<void*>(v);
}

template <typename T>
const void* convert_for_fmt(const T* const v) noexcept
{
  return reinterpret_cast<const void*>(v);
}

}  // namespace internal

template <typename... KernelArgs, typename... Args>
inline void launch_kernel(void (*kernel)(KernelArgs...),
                          const dim3& grid_dim,
                          const dim3& block_dim,
                          std::size_t shared_mem,
                          DeviceStream stream,
                          Args&&... args)
{
  static_assert(sizeof...(KernelArgs) == sizeof...(Args),
                "Number of arguments provided to launch_kernel does not match "
                "the number of expected kernel arguments");
  static_assert(
    meta::tlist::FoldlTL<
      meta::And,
      std::bool_constant<true>,
      meta::tlist::MapTL<
        internal::is_convertible,
        meta::tlist::ZipTL<meta::TL<KernelArgs...>, meta::TL<Args...>>>>::value,
    "Provided kernel arguments are not convertible to formal arguments");

  // Check grid and block dimensions.
  H2_ASSERT_DEBUG(grid_dim.x <= max_grid_x,
                  "Grid dimension x (",
                  grid_dim.x,
                  ") exceeds maximum (",
                  max_grid_x,
                  ")");
  H2_ASSERT_DEBUG(
    grid_dim.x > 0, "Grid dimension x (", grid_dim.x, ") must be > 0");
  H2_ASSERT_DEBUG(grid_dim.y <= max_grid_y,
                  "Grid dimension y (",
                  grid_dim.y,
                  ") exceeds maximum (",
                  max_grid_y,
                  ")");
  H2_ASSERT_DEBUG(
    grid_dim.y > 0, "Grid dimension y (", grid_dim.y, ") must be > 0");
  H2_ASSERT_DEBUG(grid_dim.z <= max_grid_z,
                  "Grid dimension z (",
                  grid_dim.z,
                  ") exceeds maximum (",
                  max_grid_z,
                  ")");
  H2_ASSERT_DEBUG(
    grid_dim.z > 0, "Grid dimension z (", grid_dim.z, ") must be > 0");
  H2_ASSERT_DEBUG(block_dim.x <= max_block_x,
                  "Block dimension x (",
                  block_dim.x,
                  ") exceeds maximum (",
                  max_block_x,
                  ")");
  H2_ASSERT_DEBUG(
    block_dim.x > 0, "Block dimension x (", block_dim.x, ") must be > 0");
  H2_ASSERT_DEBUG(block_dim.y <= max_block_y,
                  "Block dimension y (",
                  block_dim.y,
                  ") exceeds maximum (",
                  max_block_y,
                  ")");
  H2_ASSERT_DEBUG(
    block_dim.y > 0, "Block dimension y (", block_dim.y, ") must be > 0");
  H2_ASSERT_DEBUG(block_dim.z <= max_block_z,
                  "Block dimension z (",
                  block_dim.z,
                  ") exceeds maximum (",
                  max_block_z,
                  ")");
  H2_ASSERT_DEBUG(
    block_dim.z > 0, "Block dimension z (", block_dim.z, ") must be > 0");
  H2_ASSERT_DEBUG(block_dim.x * block_dim.y * block_dim.z
                    <= max_threads_per_block,
                  "Total threads in a block (",
                  block_dim.x * block_dim.y * block_dim.z,
                  ") exceeds maximum (",
                  max_threads_per_block,
                  ")");

  H2_GPU_TRACE("launch_kernel(kernel={} ("
                 + meta::tlist::print(meta::TL<KernelArgs...>{})
                 + "), grid_dim=({}, {}, {}), block_dim=({}, "
                   "{}, {}), shared_mem={}, stream={}, args=( "
                 + (((void) args, std::string("{} ")) + ...) + "))",
               (void*) kernel,
               grid_dim.x,
               grid_dim.y,
               grid_dim.z,
               block_dim.x,
               block_dim.y,
               block_dim.z,
               shared_mem,
               (void*) stream,
               internal::convert_for_fmt(std::forward<Args>(args))...);
  launch_kernel_internal(kernel,
                         grid_dim,
                         block_dim,
                         shared_mem,
                         stream,
                         std::forward<Args>(args)...);
}

}  // namespace gpu
}  // namespace h2
