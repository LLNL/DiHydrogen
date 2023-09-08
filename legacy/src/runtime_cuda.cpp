#include "distconv/runtime_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"

#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

namespace distconv {
namespace internal {

CUDADeviceMemoryPool::CUDADeviceMemoryPool()
{}
CUDADeviceMemoryPool::~CUDADeviceMemoryPool() {}

void *CUDADeviceMemoryPool::get(size_t size, cudaStream_t st) {
  void *p = nullptr;
  cudaError_t err =
      h2::gpu::default_cub_allocator().DeviceAllocate(&p, size, st);
  if (err != cudaSuccess) {
    size_t available;
    size_t total;
    DISTCONV_CHECK_CUDA(cudaMemGetInfo(&available, &total));
    available /= (1024 * 1024);
    total /= (1024 * 1024);
    util::PrintStreamError()
        << "Allocation of " << size
        << " bytes (" << size / 1024.0 / 1024.0 << " MB) failed. "
        << available << " MB available out of " << total << " MB.";
    std::abort();
  }
  assert_always(p);
  return p;
}

void CUDADeviceMemoryPool::release(void *p) {
  DISTCONV_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceFree(p));
}

size_t CUDADeviceMemoryPool::get_max_allocatable_size(size_t limit) {
  size_t bin_growth = h2::gpu::default_cub_allocator().bin_growth;
  size_t x = std::log(limit) / std::log(bin_growth);
  size_t max_allowed_size = std::pow(bin_growth, x);
  return max_allowed_size;
}

RuntimeCUDA *RuntimeCUDA::m_instance = nullptr;

RuntimeCUDA::RuntimeCUDA() {
  for (int i = 0; i < m_num_events; ++i) {
    DISTCONV_CHECK_CUDA(cudaEventCreateWithFlags(
        &m_events[i], cudaEventDisableTiming));
  }
}

RuntimeCUDA &RuntimeCUDA::get_instance() {
  if (m_instance == nullptr) {
    m_instance = new RuntimeCUDA();
  }
  return *m_instance;
}
#if 0
PinnedMemoryPool &RuntimeCUDA::get_pinned_memory_pool() {
  return get_instance().m_pmp;
}
#endif
CUDADeviceMemoryPool &RuntimeCUDA::get_device_memory_pool() {
  return get_instance().m_dmp;
}

cudaEvent_t &RuntimeCUDA::get_event(int idx) {
  assert_always(idx < m_num_events);
  return get_instance().m_events[idx];
}

} // namespace internal
} // namespace distconv
