#pragma once
#ifndef H2_INCLUDE_H2_GPU_ROCM_MEMORY_UTILS_HPP_INCLUDED
#define H2_INCLUDE_H2_GPU_ROCM_MEMORY_UTILS_HPP_INCLUDED

/** @file
 *
 *  Thin wrappers around hipMem{cpy,set} functions. These are here so
 *  they can be inlined if possible.
 */

#include "h2/gpu/logger.hpp"
#include "h2/gpu/runtime.hpp"
#include "h2_config.hpp"

#include <hip/hip_runtime.h>

namespace hipcub
{
class CachingDeviceAllocator;
} // namespace hipcub

namespace h2
{
namespace gpu
{

using RawCUBAllocType = hipcub::CachingDeviceAllocator;

inline void mem_copy(void* const dst, void const* const src, size_t const bytes)
{
    H2_GPU_INFO("hipMemcpy(dst={}, src={}, bytes={}, kind=hipMemcpyDefault)",
                dst,
                src,
                bytes);
    H2_CHECK_HIP(hipMemcpy(dst, src, bytes, hipMemcpyDefault));
}

inline void
mem_copy(void* const dst, void const* const src, size_t const bytes, DeviceStream const stream)
{
    H2_GPU_INFO("hipMemcpyAsync(dst={}, src={}, bytes={}, "
                "kind=hipMemcpyDefault, stream={})",
                dst,
                src,
                bytes,
                (void*) stream);
    H2_CHECK_HIP(hipMemcpyAsync(dst, src, bytes, hipMemcpyDefault, stream));
}

inline void mem_zero(void* mem, size_t bytes)
{
    H2_GPU_INFO("hipMemset(mem={}, value=0x0, bytes={})", mem, bytes);
    H2_CHECK_HIP(hipMemset(mem, 0x0, bytes));
}

inline void mem_zero(void* mem, size_t bytes, DeviceStream stream)
{
    H2_GPU_INFO("hipMemsetAsync(mem={}, value=0x0, bytes={}, stream={})",
                mem,
                bytes,
                (void*) stream);
    H2_CHECK_HIP(hipMemsetAsync(mem, 0x0, bytes, stream));
}

} // namespace gpu
} // namespace h2
#endif // H2_INCLUDE_H2_GPU_ROCM_MEMORY_UTILS_HPP_INCLUDED
