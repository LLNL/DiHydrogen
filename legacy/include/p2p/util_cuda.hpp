#pragma once

#include "distconv_config.hpp"

#include <cuda.h>

#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <vector>

#include <cuda_runtime_api.h>

#define P2P_CHECK_CUDA_ALWAYS(cuda_call)                                       \
  do                                                                           \
  {                                                                            \
    const cudaError_t cuda_status = cuda_call;                                 \
    if (cuda_status != cudaSuccess)                                            \
    {                                                                          \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n";  \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";         \
      cudaDeviceReset();                                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#ifdef P2P_DEBUG
#define P2P_CHECK_CUDA(cuda_call) P2P_CHECK_CUDA_ALWAYS(cuda_call)
#else
#define P2P_CHECK_CUDA(cuda_call) cuda_call
#endif

#define P2P_CHECK_CUDA_DRV_ALWAYS(cuda_call)                                   \
  do                                                                           \
  {                                                                            \
    const CUresult cuda_status = cuda_call;                                    \
    if (cuda_status != CUDA_SUCCESS)                                           \
    {                                                                          \
      const char* err_msg;                                                     \
      cuGetErrorString(cuda_status, &err_msg);                                 \
      std::cerr << "CUDA driver error: " << err_msg << "\n";                   \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";         \
      cudaDeviceReset();                                                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#ifdef P2P_DEBUG
#define P2P_CHECK_CUDA_DRV(cuda_call) P2P_CHECK_CUDA_DRV_ALWAYS(cuda_call)
#else
#define P2P_CHECK_CUDA_DRV(cuda_call) cuda_call
#endif

namespace p2p
{
namespace util
{

inline bool is_stream_mem_enabled()
{
  CUdevice dev;
  P2P_CHECK_CUDA_DRV_ALWAYS(cuCtxGetDevice(&dev));
  int attr;
  // There was an API change to these in CUDA 11.7, and the flag to check
  // for support changed (to have _V1) in CUDA 12. But as of CUDA 12,
  // these are enabled by default, so we do not need to check.
#if CUDA_VERSION >= 12000
  attr = 1;
#else
  P2P_CHECK_CUDA_DRV_ALWAYS(cuDeviceGetAttribute(
    &attr, CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS, dev));
#endif
  return attr;
}

class PinnedMemoryPool
{
public:
  PinnedMemoryPool();
  ~PinnedMemoryPool();

  void* get(size_t size);
  void release(void* p);

private:
  int m_bin_growth;
  int m_min_bin;
  int m_max_bin;
  std::vector<std::list<void*>> m_bins;
  std::map<void*, size_t> m_mem_map;
  std::mutex m_mutex;
  void setup_bins();
  int find_bin(size_t size);
  void* get_from_bin(int bin_idx);
  void deallocate_all_chunks();
};

class EventPool
{
public:
  EventPool(int num_events = 10, int expansion = 10);
  ~EventPool();

  cudaEvent_t get();
  void release(cudaEvent_t e);
  void expand();

private:
  int m_expansion;
  std::list<cudaEvent_t> m_events;
  std::mutex m_mutex;

  static void expand_list(std::list<cudaEvent_t>& list, int num_events);
};

size_t get_total_memory();
size_t get_available_memory();

}  // namespace util
}  // namespace p2p
