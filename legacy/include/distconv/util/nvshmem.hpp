#pragma once

#include "distconv_config.hpp"
#include "distconv/util/util_mpi.hpp"

#include <memory>

#ifdef DISTCONV_HAS_NVSHMEM
#ifndef NVSHMEM_TARGET
#define NVSHMEM_TARGET
#endif
#include "nvshmem.h"
#include "nvshmemx.h"
#endif // DISTCONV_HAS_NVSHMEM

#define DISTCONV_CHECK_NVSHMEM(call)                                    \
  do {                                                                  \
    const int status = call;                                            \
    if (status != 0) {                                                  \
      std::cerr << "NVSHMEM error: " << status << std::endl;            \
      std::cerr << "Error at " << __FILE__ << ":"                       \
                << __LINE__ << std::endl;                               \
      nvshmem_finalize();                                               \
      abort();                                                          \
    }                                                                   \
  } while (0)

namespace distconv {
namespace util {
namespace nvshmem {

#ifdef DISTCONV_HAS_NVSHMEM
void initialize(MPI_Comm comm);
void finalize();
void barrier();
void launch_barrier(cudaStream_t s);

enum class SyncType {NONE, FENCE, QUIET};

struct PairwiseSyncDevice {
  using CounterType = long;

  PairwiseSyncDevice(CounterType *local_counter,
                     CounterType *shmem_counter):
      m_local_counter(local_counter), m_shmem_counter(shmem_counter) {}

  ~PairwiseSyncDevice() = default;

#ifdef __CUDACC__
  __device__ __forceinline__ void notify(int peer, SyncType sync_type) {
    if (sync_type == SyncType::FENCE) {
      nvshmem_fence();
    } else if (sync_type == SyncType::QUIET) {
      nvshmem_quiet();
    }

    const auto counter = *m_local_counter;

    nvshmem_long_p((CounterType*)m_shmem_counter, counter, peer);
  }

  __device__ __forceinline__ void wait() {
    const auto counter = *m_local_counter;
    nvshmem_wait_until(m_shmem_counter, NVSHMEM_CMP_GE, counter);
  }

  __device__ __forceinline__ void inc_counter() {
    ++(*m_local_counter);
  }

  __device__ __forceinline__ void sync(int peer, bool do_notify, bool do_wait,
                                       SyncType sync_type) {
    if (do_notify) {
      notify(peer, sync_type);
    }
    if (do_wait) {
      wait();
    }
    inc_counter();
  }
#endif

  CounterType *m_local_counter;
  volatile CounterType *m_shmem_counter;
};

struct PairwiseSync {
  using CounterType = PairwiseSyncDevice::CounterType;
 public:
  PairwiseSync(): m_local_counter(nullptr), m_shmem_counter(nullptr) {}
  void alloc_buffers();
  void sync(int peer, bool notify, bool wait,
            SyncType sync_type, cudaStream_t stream);
  void notify(int peer, SyncType sync_type, cudaStream_t stream);
  void wait(cudaStream_t stream);
  void inc_counter(cudaStream_t stream);
  PairwiseSyncDevice get_for_device();
 private:
  std::shared_ptr<CounterType> m_local_counter;
  std::shared_ptr<CounterType> m_shmem_counter;
};

struct SyncArrayDevice {
  using CounterType = long;

  SyncArrayDevice(CounterType *local_counter,
                  CounterType *shmem_counter):
      m_local_counter(local_counter), m_shmem_counter(shmem_counter) {}

  ~SyncArrayDevice() = default;

#ifdef __CUDACC__
  __device__ __forceinline__ void notify(int peer, SyncType sync_type, int idx) {
    if (sync_type == SyncType::FENCE) {
      nvshmem_fence();
    } else if (sync_type == SyncType::QUIET) {
      nvshmem_quiet();
    }

#if 1
    const auto counter = m_local_counter[idx];
    nvshmem_long_p((CounterType*)(m_shmem_counter + idx), counter, peer);
#else
    nvshmem_long_put_nbi((CounterType*)(m_shmem_counter + idx),
                         m_local_counter + idx, 1, peer);
#endif
  }

  __device__ __forceinline__ void wait(int idx) {
    const auto counter = m_local_counter[idx];
    nvshmem_wait_until(m_shmem_counter + idx, NVSHMEM_CMP_GE, counter);
  }

  __device__ __forceinline__ void inc_counter(int idx) {
    ++m_local_counter[idx];
  }

  __device__ __forceinline__ void sync(int peer, bool do_notify, bool do_wait,
                                       SyncType sync_type, int idx) {
    if (do_notify) {
      notify(peer, sync_type, idx);
    }
    if (do_wait) {
      wait(idx);
    }
    inc_counter(idx);
  }
#endif

  CounterType *m_local_counter;
  volatile CounterType *m_shmem_counter;
};

struct SyncArray {
  using CounterType = SyncArrayDevice::CounterType;
 public:
  SyncArray(size_t size): m_local_counter(nullptr), m_shmem_counter(nullptr),
                          m_size(size) {}
  void alloc_counters();
  void init_counters();
  void ensure_size(size_t size);
  void sync(int peer, bool notify, bool wait,
            SyncType sync_type, int idx, cudaStream_t stream);
  void notify(int peer, SyncType sync_type, int idx, cudaStream_t stream);
  void wait(int idx, cudaStream_t stream);
  void inc_counter(int idx, cudaStream_t stream);
  SyncArrayDevice get_for_device();
 private:
  std::shared_ptr<CounterType> m_local_counter;
  std::shared_ptr<CounterType> m_shmem_counter;
  size_t m_size;
};

#ifdef __NVCC__
#define DEFINE_PUT(TYPE)                                                \
  inline __device__ void put(TYPE *dest, const TYPE *source,            \
                             size_t nelems, int pe) {                   \
    nvshmem_##TYPE##_put(dest, source, nelems, pe);                    \
  }                                                                     \
  inline __device__ void put(TYPE##2 *dest,                             \
                             const TYPE##2 *source,                     \
                             size_t nelems, int pe) {                   \
    nvshmem_##TYPE##_put((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }                                                                     \
  inline __device__ void put_nbi(TYPE *dest, const TYPE *source,        \
                                 size_t nelems, int pe) {               \
    nvshmem_##TYPE##_put_nbi(dest, source, nelems, pe);                \
  }                                                                     \
  inline __device__ void put_nbi(TYPE##2 *dest,                         \
                                 const TYPE##2 *source,                 \
                                 size_t nelems, int pe) {               \
    nvshmem_##TYPE##_put_nbi((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }                                                                     \
  inline __device__ void put_block(TYPE *dest, const TYPE *source,      \
                                   size_t nelems, int pe) {             \
    nvshmemx_##TYPE##_put_block(dest, source, nelems, pe);              \
  }                                                                     \
  inline __device__ void put_block(TYPE##2 *dest,                       \
                                   const TYPE##2 *source,               \
                                   size_t nelems, int pe) {             \
    nvshmemx_##TYPE##_put_block((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }                                                                     \
  inline __device__ void put_nbi_block(TYPE *dest, const TYPE *source,  \
                                       size_t nelems, int pe) {         \
    nvshmemx_##TYPE##_put_nbi_block(dest, source, nelems, pe);          \
  }                                                                     \
  inline __device__ void put_nbi_block(TYPE##2 *dest,                   \
                                       const TYPE##2 *source,           \
                                       size_t nelems, int pe) {         \
    nvshmemx_##TYPE##_put_nbi_block((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }
DEFINE_PUT(float)
DEFINE_PUT(double)
DEFINE_PUT(int)
DEFINE_PUT(long)
#undef DEFINE_PUT

#endif // __NVCC__

#define DEFINE_SUM_TO_ALL(TYPE)                                         \
  inline void sum_to_all_on_stream(TYPE *target, const TYPE *source, int nreduce, \
                                   int PE_start, int logPE_stride, int PE_size, \
                                   TYPE *pWrk, long *pSync, cudaStream_t s) { \
    nvshmemx_##TYPE##_sum_to_all_on_stream(                             \
        target, source, nreduce, PE_start,                              \
        logPE_stride, PE_size, pWrk, pSync, s);                         \
  }
DEFINE_SUM_TO_ALL(float)
DEFINE_SUM_TO_ALL(double)
DEFINE_SUM_TO_ALL(int)
DEFINE_SUM_TO_ALL(long)
#undef DEFINE_SUM_TO_ALL

#else // DISTCONV_HAS_NVSHMEM
inline void initialize(MPI_Comm comm) {}
inline void finalize() {}
inline void barrier() {}
#endif // DISTCONV_HAS_NVSHMEM

} // namespace nvshmem
} // namespace util
} // namespace distconv
