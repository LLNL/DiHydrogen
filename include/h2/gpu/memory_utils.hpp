////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 *  The public basic memory API that is exposed for user and library
 *  consumption. These are lightweight wrappers around HIP or CUDA
 *  runtime functions. They are accessible in the h2::gpu namespace.
 *
 *  struct MemInfo
 *  {
 *    size_t free;
 *    size_t total;
 *  };
 *  MemInfo mem_info();
 *
 *  typedef {cub,hipcub}::CachingDeviceAllocator RawCUBAllocType;
 *  RawCUBAllocType& default_cub_allocator();
 *
 *  void mem_copy(void* dst, void const* src, size_t bytes);
 *  void mem_copy(void* dst, void const* src, size_t bytes,
 *                DeviceStream stream);
 *
 *  void mem_zero(void* mem, size_t bytes);
 *  void mem_zero(void* mem, size_t bytes, DeviceStream stream);
 *
 *  template <typename T>
 *  void mem_copy(T* dst, T const* src, size_t n_elmts);
 *  template <typename T>
 *  void mem_copy(T* dst, T const* src, size_t n_elmts,
 *                DeviceStream stream);
 *
 *  template <typename T>
 *  void mem_zero(T* mem, size_t n_elmts);
 *  template <typename T>
 *  void mem_zero(T* mem, size_t n_elmts, DeviceStream stream);
 *
 */

#include "h2_config.hpp"

#include "runtime.hpp"

#if H2_HAS_CUDA
#include "cuda/memory_utils.hpp"
#elif H2_HAS_ROCM
#include "rocm/memory_utils.hpp"
#endif

namespace h2
{
namespace gpu
{

/** @brief The default CUB(-like) allocator used in H2.
 *
 *  This is essentially a {cub,hipcub}::CachingDeviceAllocator that
 *  has been modified to allow non-integer growth factors.
 */
RawCUBAllocType& default_cub_allocator();

/** @brief The default growth factor for a new CUB allocator.
 *
 *  Environment variable: H2_CUB_BIN_GROWTH
 *  Default value: 2U
 */
unsigned int cub_growth_factor() noexcept;

/** @brief The default smallest bin size for a new CUB allocator.
 *
 *  Environment variable: H2_CUB_MIN_BIN
 *  Default value: 1U
 */
unsigned int cub_min_bin() noexcept;

/** @brief The default largest bin size for a new CUB allocator.
 *
 *  Environment variable: H2_CUB_MAX_BIN
 *  Default value: no limit
 */
unsigned int cub_max_bin() noexcept;

/** @brief The default maximum size of a new CUB allocator.
 *
 *  Environment variable: H2_CUB_MAX_CACHED_SIZE
 *  Default value: no limit
 */
size_t cub_max_cached_size() noexcept;

/** @brief The default debugging flag for a new CUB allocator.
 *
 *  Environment variable: H2_CUB_DEBUG
 *  Default value: false
 */
bool cub_debug() noexcept;

/** @brief Create a new caching allocator.
 *
 *  @todo Expand the argument list to match the underlying ctor. This
 *        will facilitate downstream integrations and/or better
 *        environment variable configuration.
 */
RawCUBAllocType make_allocator(unsigned int const gf = cub_growth_factor(),
                               unsigned int const min = cub_min_bin(),
                               unsigned int const max = cub_max_bin(),
                               size_t const max_cached = cub_max_cached_size(),
                               bool const debug = cub_debug());

template <typename T>
inline void mem_copy(T* dst, T const* src)
{
  mem_copy(reinterpret_cast<void*>(dst),
           reinterpret_cast<void const*>(src),
           sizeof(T));
}

template <typename T>
inline void mem_copy(T* dst, T const* src, size_t n_elmts)
{
  mem_copy(reinterpret_cast<void*>(dst),
           reinterpret_cast<void const*>(src),
           n_elmts * sizeof(T));
}

template <typename T>
inline void mem_copy(T* dst, T const* src, size_t n_elmts, DeviceStream stream)
{
  mem_copy(reinterpret_cast<void*>(dst),
           reinterpret_cast<void const*>(src),
           n_elmts * sizeof(T),
           stream);
}

template <typename T>
inline void mem_zero(T* mem, size_t n_elmts)
{
  mem_zero(reinterpret_cast<void*>(mem), n_elmts * sizeof(T));
}

template <typename T>
inline void mem_zero(T* mem, size_t n_elmts, DeviceStream stream)
{
  mem_zero(reinterpret_cast<void*>(mem), n_elmts * sizeof(T), stream);
}

}  // namespace gpu
}  // namespace h2
