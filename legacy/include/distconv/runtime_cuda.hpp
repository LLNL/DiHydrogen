#pragma once

#include <cuda_runtime.h>
#include "cub/util_allocator.cuh"

namespace distconv {
namespace internal {

class CUDADeviceMemoryPool {
 public:
  CUDADeviceMemoryPool();
  ~CUDADeviceMemoryPool();
  void *get(size_t size, cudaStream_t st);
  void release(void *p, cudaStream_t st);
  size_t get_max_allocatable_size(size_t limit);
};

class RuntimeCUDA {
 public:
  static CUDADeviceMemoryPool &get_device_memory_pool();
  static cudaEvent_t &get_event(int idx=0);

 protected:
  static RuntimeCUDA *m_instance;
  //PinnedMemoryPool m_pmp;
  CUDADeviceMemoryPool m_dmp;
  static constexpr int m_num_events = 2;
  cudaEvent_t m_events[m_num_events];

  RuntimeCUDA();
  static RuntimeCUDA &get_instance();
};

} // namespace internal
} // namespace distconv
