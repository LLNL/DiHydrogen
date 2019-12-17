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

    const long counter = *m_local_counter;

    nvshmem_long_p((long*)m_shmem_counter, counter, peer);
  }

  __device__ __forceinline__ void wait() {
    const long counter = *m_local_counter;
    nvshmem_wait_until(m_shmem_counter, NVSHMEM_CMP_GE, counter);
  }

  __device__ __forceinline__ void inc_counter() {
    ++(*m_local_counter);
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

#ifdef __NVCC__
#define DEFINE_PUT_BLOCK(TYPE)                                          \
  inline __device__ void put_block(TYPE *dest, const TYPE *source,      \
                                   size_t nelems, int pe) {             \
    nvshmemx_##TYPE##_put_block(dest, source, nelems, pe);              \
  }                                                                     \
  inline __device__ void put_block(TYPE##2 *dest,                       \
                                   const TYPE##2 *source,               \
                                   size_t nelems, int pe) {             \
    nvshmemx_##TYPE##_put_block((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }

DEFINE_PUT_BLOCK(float)
DEFINE_PUT_BLOCK(double)
DEFINE_PUT_BLOCK(int)
DEFINE_PUT_BLOCK(long)
#undef DEFINE_PUT_BLOCK

#define DEFINE_PUT_NBI_BLOCK(TYPE)                                      \
  inline __device__ void put_nbi_block(TYPE *dest, const TYPE *source,  \
                                       size_t nelems, int pe) {         \
    nvshmemx_##TYPE##_put_nbi_block(dest, source, nelems, pe);          \
  }                                                                     \
  inline __device__ void put_nbi_block(TYPE##2 *dest,                   \
                                       const TYPE##2 *source,           \
                                       size_t nelems, int pe) {         \
    nvshmemx_##TYPE##_put_nbi_block((TYPE*)dest, (const TYPE*)source, nelems * 2, pe); \
  }

DEFINE_PUT_NBI_BLOCK(float)
DEFINE_PUT_NBI_BLOCK(double)
DEFINE_PUT_NBI_BLOCK(int)
DEFINE_PUT_NBI_BLOCK(long)
#undef DEFINE_PUT_NBI_BLOCK


#endif // __NVCC__

#else
inline void initialize(MPI_Comm comm) {}
inline void finalize() {}
inline void barrier() {}
#endif

} // namespace nvshmem
} // namespace util
} // namespace distconv
