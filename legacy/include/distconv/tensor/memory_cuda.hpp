#pragma once

#include "distconv/tensor/stream.hpp"
#include "distconv/tensor/stream_cuda.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/runtime_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/util/nvshmem.hpp"
#endif

#include <cuda_runtime.h>

#define TENSOR_CHECK_CUDA(cuda_call)                                    \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)


namespace distconv {
namespace tensor {

struct CUDAAllocator {
  static void allocate(void *&p, size_t &pitch,
                       size_t size, size_t ldim)  {
    DISTCONV_CUDA_MALLOC(&p, size);
    pitch = ldim;
  }
  static void deallocate(void *p)  {
    assert_always(p != nullptr);
    TENSOR_CHECK_CUDA(cudaFree(p));
  }
  static void copy(void *dst, const void *src,
                   size_t size, cudaStream_t stream=0) {
    TENSOR_CHECK_CUDA(cudaMemcpyAsync(dst, src, size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
  }
  static void memset(void *p, size_t pitch, int v,
                     size_t size, size_t,
                     cudaStream_t stream=0) {
    TENSOR_CHECK_CUDA(cudaMemsetAsync(p, v, size, stream));
  }
  static void copyin(void *dst, const void *src,
                     size_t size, size_t, size_t,
                     cudaStream_t stream=0) {
    TENSOR_CHECK_CUDA(cudaMemcpyAsync(dst, src, size,
                                      cudaMemcpyHostToDevice,
                                      stream));
  }
  static void copyout(void *dst, const void *src,
                      size_t size, size_t, size_t,
                      cudaStream_t stream=0) {
    TENSOR_CHECK_CUDA(cudaMemcpyAsync(dst, src, size,
                                      cudaMemcpyDeviceToHost,
                                      stream));
    TENSOR_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
};

struct CUDAPitchedAllocator {
  static void allocate(void *&p, size_t &pitch,
                       size_t size, size_t ldim)  {
    TENSOR_CHECK_CUDA(cudaMallocPitch(&p, &pitch, ldim, size/ldim));
  }
  static void deallocate(void *p)  {
    TENSOR_CHECK_CUDA(cudaFree(p));
  }
  static void memset(void *p, size_t pitch, int v,
                     size_t size, size_t ldim,
                     cudaStream_t stream=0) {
    TENSOR_CHECK_CUDA(cudaMemset2DAsync(p, pitch, v, ldim, size/ldim,
                                        stream));
    TENSOR_CHECK_CUDA(cudaStreamSynchronize(stream));
  }
  static void copy(void *dst, const void *src,
                   size_t size) {
    TENSOR_CHECK_CUDA(cudaMemcpy(dst, src, size,
                                 cudaMemcpyDeviceToDevice));
  }
  static void copyin(void *dst, const void *src,
                     size_t size, size_t pitch,
                     size_t ldim) {
    TENSOR_CHECK_CUDA(cudaMemcpy2D(dst, pitch,
                                   src, ldim,
                                   ldim, size / ldim,
                                   cudaMemcpyHostToDevice));
  }
  static void copyout(void *dst, const void *src,
                      size_t size, size_t pitch,
                      size_t ldim) {
    TENSOR_CHECK_CUDA(cudaMemcpy2D(dst, ldim,
                                   src, pitch,
                                   ldim, size / ldim,
                                   cudaMemcpyDeviceToHost));
  }
};

struct CUDAHostPooledAllocator {
  static void allocate(void *&p, size_t &pitch,
                       size_t size, size_t ldim)  {
    auto &x = internal::RuntimeCUDA::get_pinned_memory_pool();
    p = x.get(size);
    pitch = ldim;
  }
  static void deallocate(void *p)  {
    auto &x = internal::RuntimeCUDA::get_pinned_memory_pool();
    x.release(p);
  }
  static void memset(void *p, size_t pitch, int v,
                     size_t size, size_t,
                     int stream=0) {
    std::memset(p, v, size);
  }
  static void copyin(void *dst, const void *src,
                     size_t real_size, size_t, size_t) {
    std::memcpy(dst, src, real_size);
  }
  static void copyout(void *dst, const void *src,
                      size_t real_size, size_t, size_t) {
    std::memcpy(dst, src, real_size);
  }
};


template <typename AllocDst, typename AllocSrc,
          typename StreamType=DefaultStream>
inline typename std::enable_if<
  (std::is_same<AllocDst, CUDAAllocator>::value ||
   std::is_same<AllocDst, CUDAPitchedAllocator>::value ||
   std::is_same<AllocDst, CUDAHostPooledAllocator>::value ||
   std::is_same<AllocDst, BaseAllocator>::value) &&
  (std::is_same<AllocSrc, CUDAAllocator>::value ||
   std::is_same<AllocSrc, CUDAPitchedAllocator>::value ||
   std::is_same<AllocSrc, CUDAHostPooledAllocator>::value ||
   std::is_same<AllocSrc, BaseAllocator>::value) &&
  !(std::is_same<AllocSrc, BaseAllocator>::value &&
    std::is_same<AllocDst, BaseAllocator>::value), int>::type
Copy(Memory<AllocDst> &dst,
     const Memory<AllocSrc> &src,
     size_t x_len, size_t y_len,
     size_t x_dst_offset, size_t y_dst_offset,
     size_t x_src_offset, size_t y_src_offset,
     StreamType stream=DefaultStream::value) {
  size_t src_offset = x_src_offset + src.get_pitch() * y_src_offset;
  size_t dst_offset = x_dst_offset + dst.get_pitch() * y_dst_offset;
  void *dst_ptr = (char*)dst.get() + dst_offset;
  const void *src_ptr = (char*)src.get() + src_offset;
  cudaStream_t s = get_cuda_stream(stream);
  if (dst.get_pitch() == x_len && src.get_pitch() == x_len) {
    // Just use cudaMemcpy when possible
    TENSOR_CHECK_CUDA(cudaMemcpyAsync(dst_ptr, src_ptr, x_len * y_len,
                                      cudaMemcpyDefault, s));
  } else {
    TENSOR_CHECK_CUDA(cudaMemcpy2DAsync(dst_ptr, dst.get_pitch(),
                                        src_ptr, src.get_pitch(),
                                        x_len, y_len, cudaMemcpyDefault,
                                        s));
  }
  return 0;
}


template <>
struct Stream<CUDAAllocator> {
  using type = cudaStream_t;
  static constexpr type default_value = 0;
};

template <>
struct Stream<CUDAPitchedAllocator> {
  using type = cudaStream_t;
  static constexpr type default_value = 0;
};

template <>
struct Stream<CUDAHostPooledAllocator> {
  using type = int;
  static constexpr type default_value = 0;
};

#ifdef DISTCONV_HAS_NVSHMEM
struct NVSHMEMAllocator: CUDAAllocator {
  static void allocate(void *&p, size_t &pitch,
                       size_t size, size_t ldim)  {
    //nvshmem_barrier_all();
    p = nvshmem_malloc(size);
    nvshmem_barrier_all();
    if (p == nullptr) {
      util::PrintStreamError() << "NVSHMEM allocation of " << size << " bytes ("
                               << size / 1024.0 / 1024.0 / 1024.0 << " GiB) failed. ";
      TENSOR_CHECK_CUDA(cudaGetLastError());
    }
    pitch = ldim;
  }
  static void deallocate(void *p)  {
    assert_always(p != nullptr);
    nvshmem_barrier_all();
    nvshmem_free(p);
    nvshmem_barrier_all();
  }
};

template <>
struct Stream<NVSHMEMAllocator> {
  using type = cudaStream_t;
  static constexpr type default_value = 0;
};

#endif

} // namespace tensor
} // namespace distconv

#undef TENSOR_CHECK_CUDA
