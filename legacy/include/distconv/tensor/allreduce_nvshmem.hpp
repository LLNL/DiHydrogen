#pragma once

#include "distconv_config.hpp"

#ifdef DISTCONV_HAS_NVSHMEM

#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/nvshmem.hpp"

namespace distconv {
namespace tensor {

template <typename DataType>
class AllreduceNVSHMEM: public Allreduce<DataType> {
 public:
  AllreduceNVSHMEM(cudaStream_t stream):
      m_stream(stream), m_pid(nvshmem_my_pe()), m_np(nvshmem_n_pes()) {
    m_sync.alloc_buffers();
  }

  virtual ~AllreduceNVSHMEM() = default;

  using Allreduce<DataType>::allreduce;

  virtual void allreduce(const DataType *send_buf, DataType *recv_buf,
                         size_t count) override {
    if (m_np == 1) {
      copy(send_buf, recv_buf, count);
      return;
    }
    allreduce_naive(send_buf, recv_buf, count);
  }

 protected:
  cudaStream_t m_stream;
  int m_pid;
  int m_np;
  Memory<NVSHMEMAllocator> m_buf;
  util::nvshmem::PairwiseSync m_sync;

  void ensure_buffer(size_t count) {
    size_t cur_size = m_buf.get_size() / sizeof(DataType);
    if (cur_size >= count) {
      // the buffer is large enough
      return;
    }
    util::MPIPrintStreamInfo() << "Allocating NVSHMEM buffer of count " << count;
    m_buf.allocate(count * sizeof(DataType));
    m_buf.memset(0);
  }

  void allreduce_naive(const DataType *send_buf, DataType *recv_buf,
                       size_t count) {
    ensure_buffer(count);
    copy(send_buf, recv_buf, count);

    //int prev_pid = (m_pid + m_np - 1) % m_np;
    int next_pid = (m_pid + 1) % m_np;
    bool first_pe = m_pid == 0;
    bool last_pe = m_pid == m_np - 1;

    // wait
    if (!first_pe) {
      m_sync.wait(m_stream);
      // reduce
      reduce(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    }

    // put
    nvshmemx_putmem_on_stream((void*)m_buf.get(), recv_buf,
                              count * sizeof(DataType),
                              next_pid, m_stream);

    // notify
    if (last_pe) {
      // the last PE notifies the first PE, which is waiting with the
      // incremented counter
      m_sync.inc_counter(m_stream);
    }
    m_sync.notify(next_pid, util::nvshmem::SyncType::FENCE,
                  m_stream);

    // propagate
    if (!last_pe) {
      // the counter of the last pe already incremented
      m_sync.inc_counter(m_stream);
    }
    m_sync.wait(m_stream);
    // copy to return buffer
    copy(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    if (!last_pe) {
      // put
      nvshmemx_putmem_on_stream((void*)m_buf.get(), recv_buf,
                                count * sizeof(DataType),
                                next_pid, m_stream);
      // notify
      m_sync.notify(next_pid, util::nvshmem::SyncType::FENCE,
                    m_stream);
    }

    m_sync.inc_counter(m_stream);
  }

  void copy(const DataType *src, DataType *dst, size_t count);
  void reduce(const DataType *src, DataType *dst, size_t count);
};

} // namespace tensor
} // namespace distconv

#endif // DISTCONV_HAS_NVSHMEM
