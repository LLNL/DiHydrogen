#pragma once

#include "distconv_config.hpp"

#ifdef DISTCONV_HAS_NVSHMEM

#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/nvshmem.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_mpi.hpp"

#include <cmath>

namespace distconv
{
namespace tensor
{

template <typename DataType>
struct AllreduceNVSHMEMDevice
{
  AllreduceNVSHMEMDevice(int pid,
                         int np,
                         DataType* buf,
                         const util::nvshmem::SyncArrayDevice& sync)
    : m_pid(pid), m_np(np), m_buf(buf), m_sync(sync)
  {
    auto log_np = std::log2((float) m_np);
    assert_always(std::ceil(log_np) == std::floor(log_np));
    m_num_steps = log_np;
  }
  int m_pid;
  int m_np;
  int m_num_steps;
  DataType* m_buf;
  util::nvshmem::SyncArrayDevice m_sync;
#ifdef __CUDACC__
  // Assume that the root thread has the partial sum. Those held by
  // other threads are not used. Only the root thread has the valid
  // output value.
  __device__ DataType recursive_doubling_block(DataType psum,
                                               size_t num_blocks_per_entry)
  {
    const int tid = threadIdx.x;
    const int block_offset =
      (blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y)
      / num_blocks_per_entry;
    const int sync_idx = block_offset * m_num_steps;
    constexpr auto st = util::nvshmem::SyncType::NONE;

    DataType* tmp_buf =
      m_buf + (num_blocks_per_entry + m_num_steps) * block_offset;
    DataType final_sum;

    if (tid == 0)
      *tmp_buf = psum;

    // TODO: intra-grid partial reduction

    if (tid == 0)
    {
      // Inter-device reduction
      for (int i = 0; i < m_num_steps; ++i)
      {
        int peer = m_pid ^ (1 << i);
        util::nvshmem::put_nbi(tmp_buf + 1, tmp_buf, 1, peer);
        m_sync.sync(peer, true, true, st, sync_idx + i);
        tmp_buf[1] += tmp_buf[0];
        ++tmp_buf;
      }
      final_sum = *tmp_buf;
    }

    // TODO: intra-grid scatter
    return final_sum;
  }
#endif // __CUDACC__
};

template <typename DataType>
class AllreduceNVSHMEM : public Allreduce<DataType>
{
public:
  enum Algo
  {
    NAIVE,
    NATIVE,
    RECURSIVE_DOUBLING_HOST,
    RECURSIVE_DOUBLING,
    RECURSIVE_DOUBLING_BUFFERED,
    RECURSIVE_DOUBLING_BLOCK
  };
  AllreduceNVSHMEM(cudaStream_t stream, Algo algo = NAIVE)
    : m_stream(stream),
      m_algo(algo),
      m_pid(nvshmem_my_pe()),
      m_np(nvshmem_n_pes()),
      m_sync(0)
  {}

  virtual ~AllreduceNVSHMEM() = default;

  using Allreduce<DataType>::allreduce;

  virtual void
  allreduce(const DataType* send_buf, DataType* recv_buf, size_t count) override
  {
    if (m_np == 1)
    {
      copy(send_buf, recv_buf, count);
      return;
    }
    switch (m_algo)
    {
    case NAIVE: allreduce_naive(send_buf, recv_buf, count); break;
    case NATIVE: allreduce_native(send_buf, recv_buf, count); break;
    case RECURSIVE_DOUBLING_HOST:
      recursive_doubling_host(send_buf, recv_buf, count);
      break;
    case RECURSIVE_DOUBLING:
      recursive_doubling(send_buf, recv_buf, count);
      break;
    case RECURSIVE_DOUBLING_BUFFERED:
      recursive_doubling_buffered(send_buf, recv_buf, count);
      break;
    case RECURSIVE_DOUBLING_BLOCK:
      recursive_doubling_block(send_buf, recv_buf, count);
      break;
    default:
      util::MPIRootPrintStreamError() << "Unknown allreduce algorithm";
      std::abort();
    }
  }

  // Setup data buffers and sync buffers
  void recursive_doubling_block_setup(size_t count, size_t num_blocks_per_entry)
  {
    auto log_np = std::log2((float) m_np);
    assert_always(std::ceil(log_np) == std::floor(log_np));
    const int num_steps = log_np;
    ensure_buffer((num_blocks_per_entry + num_steps) * count);
    m_sync.ensure_size(num_steps * count);
  }

  // template <typename T=DataType>
  template <typename T>
  AllreduceNVSHMEMDevice<T> get_for_device()
  {
    return AllreduceNVSHMEMDevice<T>(
      m_pid, m_np, static_cast<T*>(m_buf.get()), m_sync.get_for_device());
  }

protected:
  cudaStream_t m_stream;
  Algo m_algo;
  int m_pid;
  int m_np;
  Memory<NVSHMEMAllocator> m_buf;
  util::nvshmem::SyncArray m_sync;
  Memory<NVSHMEMAllocator> m_native_sync;

  void ensure_buffer(size_t count)
  {
    size_t cur_size = m_buf.get_size() / sizeof(DataType);
    if (cur_size >= count)
    {
      // the buffer is large enough
      return;
    }
    util::MPIPrintStreamInfo()
      << "Allocating NVSHMEM buffer of count " << count;
    m_buf.allocate(count * sizeof(DataType));
    m_buf.memset(0);
  }

  void
  allreduce_naive(const DataType* send_buf, DataType* recv_buf, size_t count)
  {
    ensure_buffer(count);
    copy(send_buf, recv_buf, count);

    m_sync.ensure_size(1);

    // int prev_pid = (m_pid + m_np - 1) % m_np;
    int next_pid = (m_pid + 1) % m_np;
    bool first_pe = m_pid == 0;
    bool last_pe = m_pid == m_np - 1;
    const int counter_idx = 0;

    // wait
    if (!first_pe)
    {
      m_sync.wait(counter_idx, m_stream);
      // reduce
      reduce(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    }

    // put
    nvshmemx_putmem_on_stream((void*) m_buf.get(),
                              recv_buf,
                              count * sizeof(DataType),
                              next_pid,
                              m_stream);

    // notify
    if (last_pe)
    {
      // the last PE notifies the first PE, which is waiting with the
      // incremented counter
      m_sync.inc_counter(counter_idx, m_stream);
    }
    m_sync.notify(
      next_pid, util::nvshmem::SyncType::FENCE, counter_idx, m_stream);

    // propagate
    if (!last_pe)
    {
      // the counter of the last pe already incremented
      m_sync.inc_counter(counter_idx, m_stream);
    }
    m_sync.wait(counter_idx, m_stream);
    // copy to return buffer
    copy(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    if (!last_pe)
    {
      // put
      nvshmemx_putmem_on_stream((void*) m_buf.get(),
                                recv_buf,
                                count * sizeof(DataType),
                                next_pid,
                                m_stream);
      // notify
      m_sync.notify(
        next_pid, util::nvshmem::SyncType::FENCE, counter_idx, m_stream);
    }

    m_sync.inc_counter(counter_idx, m_stream);
  }

  void ensure_native_sync()
  {
    using SyncType = long;
    size_t cur_size = m_native_sync.get_size() / sizeof(SyncType);
    size_t required_size = NVSHMEMI_REDUCE_SYNC_SIZE;
    if (cur_size >= required_size)
      return;
    m_native_sync.allocate(required_size * sizeof(SyncType));
  }

  void ensure_native_buffer(size_t count)
  {
    size_t required_size =
      std::max(count / 2 + 1, (size_t) NVSHMEMI_REDUCE_MIN_WRKDATA_SIZE);
    ensure_buffer(required_size);
  }

  void
  allreduce_native(const DataType* send_buf, DataType* recv_buf, size_t count)
  {
    ensure_native_sync();
    ensure_native_buffer(count);
    int pe_start = 0;
    int pe_size = nvshmem_n_pes();
    util::nvshmem::sum_to_all_on_stream(recv_buf,
                                        send_buf,
                                        count,
                                        pe_start,
                                        0,
                                        pe_size,
                                        static_cast<DataType*>(m_buf.get()),
                                        static_cast<long*>(m_native_sync.get()),
                                        m_stream);
  }

  void recursive_doubling_host(const DataType* send_buf,
                               DataType* recv_buf,
                               size_t count)
  {
    auto log_np = std::log2((float) m_np);
    assert_always(std::ceil(log_np) == std::floor(log_np));

    ensure_buffer(count);
    copy(send_buf, recv_buf, count);

    int num_steps = log_np;
    const size_t len = count * sizeof(DataType);
    // make sure there are sync objects for each stage
    m_sync.ensure_size(num_steps);
    for (int i = 0; i < num_steps; ++i)
    {
      int peer = m_pid ^ (1 << i);
      m_sync.sync(
        peer, true, true, util::nvshmem::SyncType::FENCE, i, m_stream);
      nvshmemx_putmem_on_stream(
        (void*) m_buf.get(), recv_buf, len, peer, m_stream);
      m_sync.sync(
        peer, true, true, util::nvshmem::SyncType::FENCE, i, m_stream);
      reduce(static_cast<DataType*>(m_buf.get()), recv_buf, count);
    }
  }

  void set_blocking_params(size_t count,
                           size_t& work_per_block,
                           int& block_size,
                           int& grid_size)
  {
    // default work size
    work_per_block = 1024;
    work_per_block = std::min(count, work_per_block);
    // override if set
    auto env = std::getenv("DISTCONV_RECURSIVE_DOUBLING_WORK_PER_BLOCK");
    if (env)
    {
      work_per_block = std::atoi(env);
    }

    block_size = std::max(32, (int) std::min(work_per_block, (size_t) 256));
    grid_size = (count + work_per_block - 1) / work_per_block;
  }

  void recursive_doubling(const DataType* send_buf,
                          DataType* recv_buf,
                          size_t count);
  void recursive_doubling_buffered(const DataType* send_buf,
                                   DataType* recv_buf,
                                   size_t count);

  void recursive_doubling_block(const DataType* send_buf,
                                DataType* recv_buf,
                                size_t count);

  void copy(const DataType* src, DataType* dst, size_t count);
  void reduce(const DataType* src, DataType* dst, size_t count);
};

} // namespace tensor
} // namespace distconv

#endif // DISTCONV_HAS_NVSHMEM
