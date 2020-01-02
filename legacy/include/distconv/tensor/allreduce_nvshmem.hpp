#pragma once

#include "distconv_config.hpp"

#ifdef DISTCONV_HAS_NVSHMEM

#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/nvshmem.hpp"

#include <cmath>

namespace distconv {
namespace tensor {

template <typename DataType>
class AllreduceNVSHMEM: public Allreduce<DataType> {
 public:
  enum Algo {NAIVE, RECURSIVE_DOUBLING};
  AllreduceNVSHMEM(cudaStream_t stream, Algo algo=NAIVE):
      m_stream(stream), m_algo(algo), m_pid(nvshmem_my_pe()), m_np(nvshmem_n_pes()) {
  }

  virtual ~AllreduceNVSHMEM() = default;

  using Allreduce<DataType>::allreduce;

  virtual void allreduce(const DataType *send_buf, DataType *recv_buf,
                         size_t count) override {
    if (m_np == 1) {
      copy(send_buf, recv_buf, count);
      return;
    }
    switch (m_algo) {
      case NAIVE:
        allreduce_naive(send_buf, recv_buf, count);
        break;
      case RECURSIVE_DOUBLING:
        allreduce_recursive_doubling(send_buf, recv_buf, count);
        break;
      default:
        util::MPIRootPrintStreamError() << "Unknown allreduce algorithm";
        std::abort();
    }
  }

 protected:
  cudaStream_t m_stream;
  Algo m_algo;
  int m_pid;
  int m_np;
  Memory<NVSHMEMAllocator> m_buf;
  std::vector<util::nvshmem::PairwiseSync> m_sync;

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

    if (m_sync.size() == 0) {
      m_sync.resize(1);
      m_sync[0].alloc_buffers();
    }

    //int prev_pid = (m_pid + m_np - 1) % m_np;
    int next_pid = (m_pid + 1) % m_np;
    bool first_pe = m_pid == 0;
    bool last_pe = m_pid == m_np - 1;
    auto &sync = m_sync[0];

    // wait
    if (!first_pe) {
      sync.wait(m_stream);
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
      sync.inc_counter(m_stream);
    }
    sync.notify(next_pid, util::nvshmem::SyncType::FENCE,
                  m_stream);

    // propagate
    if (!last_pe) {
      // the counter of the last pe already incremented
      sync.inc_counter(m_stream);
    }
    sync.wait(m_stream);
    // copy to return buffer
    copy(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    if (!last_pe) {
      // put
      nvshmemx_putmem_on_stream((void*)m_buf.get(), recv_buf,
                                count * sizeof(DataType),
                                next_pid, m_stream);
      // notify
      sync.notify(next_pid, util::nvshmem::SyncType::FENCE,
                    m_stream);
    }

    sync.inc_counter(m_stream);
  }

  void allreduce_recursive_doubling(const DataType *send_buf, DataType *recv_buf,
                                    size_t count) {
    auto log_np = std::log2((float)m_np);
    assert_always(std::ceil(log_np) == std::floor(log_np));

    ensure_buffer(count);
    copy(send_buf, recv_buf, count);

    int num_steps = log_np;
    const size_t len = count * sizeof(DataType);
    // make sure there are sync objects for each stage
    if (m_sync.size() < num_steps) {
      m_sync.resize(num_steps);
      for (auto &s: m_sync) {
        s.alloc_buffers();
      }
    }
    for (int i = 0; i < num_steps; ++i) {
      util::MPIPrintStreamInfo() << "Recursive doubling step: " << i;
      int peer = m_pid ^ (1 << i);
      m_sync[i].sync(peer, true, true, util::nvshmem::SyncType::FENCE,
                     m_stream);
      nvshmemx_putmem_on_stream((void*)m_buf.get(), recv_buf, len,
                                peer, m_stream);
      m_sync[i].sync(peer, true, true, util::nvshmem::SyncType::FENCE,
                     m_stream);
      reduce(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    }
  }


  void copy(const DataType *src, DataType *dst, size_t count);
  void reduce(const DataType *src, DataType *dst, size_t count);
};

} // namespace tensor
} // namespace distconv

#endif // DISTCONV_HAS_NVSHMEM
