#pragma once
#ifndef H2_INCLUDE_H2_MEMORY_GPU_MEMORY_UTILS_HPP_INCLUDED
#define H2_INCLUDE_H2_MEMORY_GPU_MEMORY_UTILS_HPP_INCLUDED

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

namespace h2
{
namespace gpu
{

struct MemInfo
{
    size_t free;
    size_t total;
};

}// namespace gpu
}// namespace h2

#if H2_HAS_CUDA
#include "cuda/memory_utils.hpp"
#elif H2_HAS_ROCM
#include "rocm/memory_utils.hpp"
#endif

// Forward-declare the {cub,hipcub}::CachingDeviceAllocator class.
namespace H2_CUB_NAMESPACE
{
class CachingDeviceAllocator;
}

namespace h2
{
namespace gpu
{

RawCUBAllocType& default_cub_allocator();

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

} // namespace gpu
} // namespace h2
#endif // H2_INCLUDE_H2_MEMORY_GPU_MEMORY_UTILS_HPP_INCLUDED
