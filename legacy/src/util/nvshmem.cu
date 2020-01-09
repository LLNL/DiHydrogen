#include "distconv/util/nvshmem.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"

namespace distconv {
namespace util {
namespace nvshmem {

#ifdef DISTCONV_HAS_NVSHMEM
void initialize(MPI_Comm comm) {
  util::MPIRootPrintStreamInfo() << "Initializing NVSHMEM with MPI";
  nvshmemx_init_attr_t attr;
  attr.mpi_comm = &comm;
  DISTCONV_CHECK_NVSHMEM(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr));
}

void finalize() {
  util::MPIRootPrintStreamInfo() << "Finalizing NVSHMEM";
  nvshmem_finalize();
}

void barrier() {
  nvshmem_barrier_all();
}

namespace internal {

__global__ void sync_pairwise_kernel(int peer,
                                     bool notify, bool wait,
                                     SyncType sync_type,
                                     PairwiseSyncDevice sync) {
  sync.sync(peer, notify, wait, sync_type);
}

__global__ void notify_kernel(int peer,
                              SyncType sync_type,
                              PairwiseSyncDevice sync) {
  sync.notify(peer, sync_type);
}

__global__ void wait_kernel(PairwiseSyncDevice sync) {
  sync.wait();
}

__global__ void inc_counter_kernel(PairwiseSyncDevice sync) {
  sync.inc_counter();
}

} // namespace internal

void PairwiseSync::sync(int peer, bool notify, bool wait,
                        SyncType sync_type, cudaStream_t stream) {
  if (peer == MPI_PROC_NULL) return;
  internal::sync_pairwise_kernel<<<1, 1, 0, stream>>>(
      peer, notify, wait, sync_type, get_for_device());
}

void PairwiseSync::notify(int peer, SyncType sync_type,
                          cudaStream_t stream) {
  if (peer == MPI_PROC_NULL) return;
  internal::notify_kernel<<<1, 1, 0, stream>>>(
      peer, sync_type, get_for_device());
}

void PairwiseSync::wait(cudaStream_t stream) {
  internal::wait_kernel<<<1, 1, 0, stream>>>(get_for_device());
}

void PairwiseSync::inc_counter(cudaStream_t stream) {
  internal::inc_counter_kernel<<<1, 1, 0, stream>>>(get_for_device());
}

void PairwiseSync::alloc_buffers() {
  if (m_shmem_counter) {
    // already allocated
    return;
  }
  CounterType *shmem_counter = static_cast<CounterType*>(
      nvshmem_malloc(sizeof(CounterType)));
  //util::MPIPrintStreamDebug() << "shmem flag: " << p;
  if (shmem_counter == nullptr) {
    util::MPIPrintStreamError() << "Allocation of shmem buffer failed";
    throw std::exception();
  }
  DISTCONV_CHECK_CUDA(cudaMemset(shmem_counter, 0, sizeof(CounterType)));
  // Make sure the memset is completed
  DISTCONV_CHECK_CUDA(cudaStreamSynchronize(0));
  barrier();
  m_shmem_counter = std::shared_ptr<CounterType>(
      shmem_counter, [](CounterType *ptr) { nvshmem_free(ptr); });

  // Setup the device counter variable
  CounterType *local_counter = nullptr;
  DISTCONV_CHECK_CUDA(cudaMalloc(&local_counter, sizeof(CounterType)));
  CounterType counter_init = 1;
  DISTCONV_CHECK_CUDA(cudaMemcpy(
      local_counter, &counter_init,
      sizeof(CounterType), cudaMemcpyHostToDevice));
  m_local_counter = std::shared_ptr<CounterType>(
      local_counter, [](CounterType *ptr) {
                       DISTCONV_CHECK_CUDA(cudaFree(ptr)); });
}

PairwiseSyncDevice PairwiseSync::get_for_device() {
  return PairwiseSyncDevice(m_local_counter.get(), m_shmem_counter.get());
}

namespace internal {

__global__ void sync_kernel(int peer, bool notify, bool wait,
                            SyncType sync_type, int idx,
                            SyncArrayDevice sync) {
  sync.sync(peer, notify, wait, sync_type, idx);
}

__global__ void notify_kernel(int peer, SyncType sync_type, int idx,
                              SyncArrayDevice sync) {
  sync.notify(peer, sync_type, idx);
}

__global__ void wait_kernel(int idx, SyncArrayDevice sync) {
  sync.wait(idx);
}

__global__ void inc_counter_kernel(int idx, SyncArrayDevice sync) {
  sync.inc_counter(idx);
}

} // namespace internal

void SyncArray::sync(int peer, bool notify, bool wait,
                     SyncType sync_type, int idx, cudaStream_t stream) {
  if (peer == MPI_PROC_NULL) return;
  internal::sync_kernel<<<1, 1, 0, stream>>>(
      peer, notify, wait, sync_type, idx, get_for_device());
}

void SyncArray::notify(int peer, SyncType sync_type, int idx,
                       cudaStream_t stream) {
  if (peer == MPI_PROC_NULL) return;
  internal::notify_kernel<<<1, 1, 0, stream>>>(
      peer, sync_type, idx, get_for_device());
}

void SyncArray::wait(int idx, cudaStream_t stream) {
  internal::wait_kernel<<<1, 1, 0, stream>>>(idx, get_for_device());
}

void SyncArray::inc_counter(int idx, cudaStream_t stream) {
  internal::inc_counter_kernel<<<1, 1, 0, stream>>>(idx, get_for_device());
}

void SyncArray::alloc_counters() {
  if (m_shmem_counter) {
    // already allocated
    return;
  }
  if (m_size == 0) {
    // nothing to allocate
    return;
  }
  CounterType *shmem_counter = static_cast<CounterType*>(
      nvshmem_malloc(sizeof(CounterType) * m_size));
  //util::MPIPrintStreamDebug() << "shmem flag: " << p;
  if (shmem_counter == nullptr) {
    util::MPIPrintStreamError() << "Allocation of shmem buffer failed";
    throw std::exception();
  }
  m_shmem_counter = std::shared_ptr<CounterType>(
      shmem_counter, [](CounterType *ptr) { nvshmem_free(ptr); });
  // Setup the device counter variable
  CounterType *local_counter = static_cast<CounterType*>(
      nvshmem_malloc(sizeof(CounterType) * m_size));
  if (shmem_counter == nullptr) {
    util::MPIPrintStreamError() << "Allocation of local buffer failed";
    throw std::exception();
  }
  m_local_counter = std::shared_ptr<CounterType>(
      local_counter, [](CounterType *ptr) { nvshmem_free(ptr); });
  init_counters();
}

void SyncArray::init_counters() {
  DISTCONV_CHECK_CUDA(cudaMemset(
      m_shmem_counter.get(), 0, sizeof(CounterType) * m_size));
  std::vector<CounterType> init(m_size,  1);
  DISTCONV_CHECK_CUDA(cudaMemcpy(
      m_local_counter.get(), init.data(),
      sizeof(CounterType) * m_size, cudaMemcpyHostToDevice));
  // Make sure the memset is completed
  DISTCONV_CHECK_CUDA(cudaStreamSynchronize(0));
  barrier();
}

void SyncArray::ensure_size(size_t size) {
  if (m_size < size) {
    m_size = size;
    alloc_counters();
  }
}

SyncArrayDevice SyncArray::get_for_device() {
  return SyncArrayDevice(m_local_counter.get(), m_shmem_counter.get());
}

#endif // DISTCONV_HAS_NVSHMEM

} // namespace nvshmem
} // namespace util
} // namespace distconv
