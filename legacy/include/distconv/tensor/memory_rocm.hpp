////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/runtime_rocm.hpp"
#include "distconv/tensor/stream.hpp"
#include "distconv/tensor/stream_rocm.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_rocm.hpp"
#include "h2/gpu/memory_utils.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/util/nvshmem.hpp"
#endif

#include <hip/hip_runtime.h>

#define TENSOR_CHECK_HIP(hip_call) DISTCONV_CHECK_HIP(hip_call)

namespace distconv
{
namespace tensor
{

// I'm not changing any of the struct names at this time. These are
// being rewritten anyway, but this saves me having to add a bunch of
// typedefs all over.
struct CUDAAllocator
{
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    DISTCONV_HIP_MALLOC(&p, size);
    pitch = ldim;
  }
  static void deallocate(void* p)
  {
    assert_always(p != nullptr);
    TENSOR_CHECK_HIP(hipFree(p));
  }
  static void
  copy(void* dst, const void* src, size_t size, hipStream_t stream = 0)
  {
    h2::gpu::mem_copy(dst, src, size, stream);
  }
  static void memset(
    void* p, size_t pitch, int v, size_t size, size_t, hipStream_t stream = 0)
  {
    TENSOR_CHECK_HIP(hipMemsetAsync(p, v, size, stream));
  }
  static void copyin(void* dst,
                     const void* src,
                     size_t size,
                     size_t,
                     size_t,
                     hipStream_t stream = 0)
  {
    h2::gpu::mem_copy(dst, src, size, stream);
  }
  static void copyout(void* dst,
                      const void* src,
                      size_t size,
                      size_t,
                      size_t,
                      hipStream_t stream = 0)
  {
    h2::gpu::mem_copy(dst, src, size, stream);
    h2::gpu::sync(stream);
  }
};

struct CUDAPitchedAllocator
{
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    TENSOR_CHECK_HIP(hipMallocPitch(&p, &pitch, ldim, size / ldim));
  }
  static void deallocate(void* p) { TENSOR_CHECK_HIP(hipFree(p)); }
  static void memset(void* p,
                     size_t pitch,
                     int v,
                     size_t size,
                     size_t ldim,
                     hipStream_t stream = 0)
  {
    TENSOR_CHECK_HIP(hipMemset2DAsync(p, pitch, v, ldim, size / ldim, stream));
    h2::gpu::sync(stream);
  }
  static void copy(void* dst, const void* src, size_t size)
  {
    h2::gpu::mem_copy(dst, src, size);
  }
  static void
  copyin(void* dst, const void* src, size_t size, size_t pitch, size_t ldim)
  {
    TENSOR_CHECK_HIP(hipMemcpy2D(
      dst, pitch, src, ldim, ldim, size / ldim, hipMemcpyHostToDevice));
  }
  static void
  copyout(void* dst, const void* src, size_t size, size_t pitch, size_t ldim)
  {
    TENSOR_CHECK_HIP(hipMemcpy2D(
      dst, ldim, src, pitch, ldim, size / ldim, hipMemcpyDeviceToHost));
  }
};

struct CUDAHostPooledAllocator
{
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    auto& x = internal::RuntimeHIP::get_pinned_memory_pool();
    p = x.get(size);
    pitch = ldim;
  }
  static void deallocate(void* p)
  {
    auto& x = internal::RuntimeHIP::get_pinned_memory_pool();
    x.release(p);
  }
  static void
  memset(void* p, size_t pitch, int v, size_t size, size_t, int stream = 0)
  {
    std::memset(p, v, size);
  }
  static void
  copyin(void* dst, const void* src, size_t real_size, size_t, size_t)
  {
    std::memcpy(dst, src, real_size);
  }
  static void
  copyout(void* dst, const void* src, size_t real_size, size_t, size_t)
  {
    std::memcpy(dst, src, real_size);
  }
};

template <typename AllocDst,
          typename AllocSrc,
          typename StreamType = DefaultStream>
inline typename std::enable_if<
  (std::is_same<AllocDst, CUDAAllocator>::value
   || std::is_same<AllocDst, CUDAPitchedAllocator>::value
   || std::is_same<AllocDst, CUDAHostPooledAllocator>::value
   || std::is_same<AllocDst, BaseAllocator>::value)
    && (std::is_same<AllocSrc, CUDAAllocator>::value
        || std::is_same<AllocSrc, CUDAPitchedAllocator>::value
        || std::is_same<AllocSrc, CUDAHostPooledAllocator>::value
        || std::is_same<AllocSrc, BaseAllocator>::value)
    && !(std::is_same<AllocSrc, BaseAllocator>::value
         && std::is_same<AllocDst, BaseAllocator>::value),
  int>::type
Copy(Memory<AllocDst>& dst,
     const Memory<AllocSrc>& src,
     size_t x_len,
     size_t y_len,
     size_t x_dst_offset,
     size_t y_dst_offset,
     size_t x_src_offset,
     size_t y_src_offset,
     StreamType stream = DefaultStream::value)
{
  size_t src_offset = x_src_offset + src.get_pitch() * y_src_offset;
  size_t dst_offset = x_dst_offset + dst.get_pitch() * y_dst_offset;
  void* dst_ptr = (char*) dst.get() + dst_offset;
  const void* src_ptr = (char*) src.get() + src_offset;
  hipStream_t s = get_gpu_stream(stream);
  if (dst.get_pitch() == x_len && src.get_pitch() == x_len)
  {
    // Just use hipMemcpy when possible
    h2::gpu::mem_copy(dst_ptr, src_ptr, x_len * y_len, s);
  }
  else
  {
    TENSOR_CHECK_HIP(hipMemcpy2DAsync(dst_ptr,
                                      dst.get_pitch(),
                                      src_ptr,
                                      src.get_pitch(),
                                      x_len,
                                      y_len,
                                      hipMemcpyDefault,
                                      s));
  }
  return 0;
}

template <>
struct Stream<CUDAAllocator>
{
  using type = hipStream_t;
  static constexpr type default_value = 0;
};

template <>
struct Stream<CUDAPitchedAllocator>
{
  using type = hipStream_t;
  static constexpr type default_value = 0;
};

template <>
struct Stream<CUDAHostPooledAllocator>
{
  using type = int;
  static constexpr type default_value = 0;
};

// This won't ever be enabled in ROCm-land. However, one of these
// years we might get HIP-SHMEM up and running. So I'll leave it in
// case it's ever useful.
#ifdef DISTCONV_HAS_NVSHMEM
struct NVSHMEMAllocator : HIPAllocator
{
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    // nvshmem_barrier_all();
    p = nvshmem_malloc(size);
    nvshmem_barrier_all();
    if (p == nullptr)
    {
      util::PrintStreamError()
        << "NVSHMEM allocation of " << size << " bytes ("
        << size / 1024.0 / 1024.0 / 1024.0 << " GiB) failed. ";
      TENSOR_CHECK_HIP(hipGetLastError());
    }
    pitch = ldim;
  }
  static void deallocate(void* p)
  {
    assert_always(p != nullptr);
    nvshmem_barrier_all();
    nvshmem_free(p);
    nvshmem_barrier_all();
  }
};

template <>
struct Stream<NVSHMEMAllocator>
{
  using type = hipStream_t;
  static constexpr type default_value = 0;
};

#endif

} // namespace tensor
} // namespace distconv
