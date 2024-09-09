#pragma once

#include "distconv/tensor/shuffle_mpi_cuda.hpp"

#include "p2p/p2p.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType>
class TensorMPICUDAShufflerP2P : public TensorMPICUDAShuffler<DataType>
{
  using TensorType = typename TensorMPICUDAShuffler<DataType>::TensorType;

public:
  TensorMPICUDAShufflerP2P(const TensorType& src_tensor,
                           const TensorType& dst_tensor,
                           p2p::P2P& p2p,
                           DataType* src_buf = nullptr,
                           DataType* dst_buf = nullptr)
    : TensorMPICUDAShuffler<DataType>(src_tensor, dst_tensor, src_buf, dst_buf),
      m_p2p(p2p)
  {
    setup_p2p_connections();
  }

  virtual ~TensorMPICUDAShufflerP2P()
  {
    m_p2p.close_addrs(m_conns, get_peer_addrs(true), this->get_num_peers());
    m_p2p.close_addrs(m_conns, get_peer_addrs(false), this->get_num_peers());
    delete[] m_conns;
    if (this->m_src_buf_passed)
    {
      DISTCONV_CHECK_CUDA(cudaFree(this->m_src_buf));
    }
    if (this->m_dst_buf_passed)
    {
      DISTCONV_CHECK_CUDA(cudaFree(this->m_dst_buf));
    }
    for (int i = 0; i < 2; ++i)
    {
      delete[] m_peer_addrs[i];
      delete[] m_peer_offsets[i];
    }
    for (int i = 0; i < this->get_num_peers(); ++i)
    {
      DISTCONV_CHECK_CUDA(cudaStreamDestroy(m_streams[i]));
    }
    delete[] m_streams;
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_ev));
  }

protected:
  p2p::P2P& m_p2p;
  p2p::P2P::connection_type* m_conns;
  void** m_peer_addrs[2];
  size_t* m_peer_offsets[2];
  cudaStream_t* m_streams;
  cudaEvent_t m_ev;

  void**& get_peer_addrs(bool is_forward)
  {
    return m_peer_addrs[is_forward ? 0 : 1];
  }

  size_t*& get_peer_offsets(bool is_forward)
  {
    return m_peer_offsets[is_forward ? 0 : 1];
  }

  void setup_peer_addresses(bool is_forward)
  {
    int num_peers = this->get_num_peers();
    get_peer_addrs(is_forward) = new void*[num_peers];
    get_peer_offsets(is_forward) = new size_t[num_peers];

    void** local_addrs = new void*[num_peers];
    size_t* local_offsets = new size_t[num_peers];
    for (int i = 0; i < num_peers; ++i)
    {
      if (this->get_recv_counts(is_forward)[this->m_peers[i]] != 0)
      {
        local_addrs[i] = get_dst_buf(is_forward);
      }
      else
      {
        // do not expose memory to the peer when nothing should be
        // received from the peer
        local_addrs[i] = nullptr;
      }
    }
    int local_offset_idx = 0;
    size_t offset = 0;
    for (int rank = 0; rank < this->m_loc.get_size(); ++rank)
    {
      if (this->get_recv_counts(is_forward)[rank] != 0)
      {
        local_offsets[local_offset_idx] = offset;
        offset += this->get_recv_counts(is_forward)[rank];
        ++local_offset_idx;
      }
      else if (this->get_send_counts(is_forward)[rank] != 0)
      {
        // in this case, the base addr is null, so the offset value is
        // not going to be used.
        local_offsets[local_offset_idx] = 0;
        ++local_offset_idx;
      }
    }

    m_p2p.exchange_addrs(m_conns,
                         local_addrs,
                         local_offsets,
                         get_peer_addrs(is_forward),
                         get_peer_offsets(is_forward),
                         num_peers);

    delete[] local_addrs;
    delete[] local_offsets;
  }

  void setup_p2p_connections()
  {
    util::MPIPrintStreamDebug() << "Setting up P2P connections for shuffling\n";
    int num_peers = this->get_num_peers();
    m_conns = new p2p::P2P::connection_type[num_peers];
    m_p2p.get_connections(this->m_peers.data(), m_conns, num_peers);

    if (!this->m_src_buf_passed)
    {
      auto buffer_size =
        TensorMPICUDAShuffler<DataType>::get_buf_size(this->m_src_local_shape);
      DISTCONV_CUDA_MALLOC(&this->m_src_buf, buffer_size);
    }
    if (!this->m_dst_buf_passed)
    {
      auto buffer_size =
        TensorMPICUDAShuffler<DataType>::get_buf_size(this->m_dst_local_shape);
      DISTCONV_CUDA_MALLOC(&this->m_dst_buf, buffer_size);
    }

    // setup peer addresses
    setup_peer_addresses(true);
    setup_peer_addresses(false);

    m_streams = new cudaStream_t[num_peers];
    for (int i = 0; i < num_peers; ++i)
    {
      DISTCONV_CHECK_CUDA(cudaStreamCreate(&m_streams[i]));
    }

    DISTCONV_CHECK_CUDA(
      cudaEventCreateWithFlags(&m_ev, cudaEventDisableTiming));
  }

  DataType* get_src_buf(bool is_forward, cudaStream_t = 0) override
  {
    return static_cast<DataType*>(is_forward ? this->m_src_buf
                                             : this->m_dst_buf);
  }

  DataType* get_dst_buf(bool is_forward, cudaStream_t = 0) override
  {
    return static_cast<DataType*>(is_forward ? this->m_dst_buf
                                             : this->m_src_buf);
  }

  void transfer(const DataType* send_buf,
                size_t send_buffer_size,
                DataType* recv_buf,
                size_t recv_buffer_size,
                bool is_forward,
                cudaStream_t stream) override
  {
    DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev, stream));
    int num_peers = this->get_num_peers();
#if 1
    for (int i = 0; i < num_peers; ++i)
    {
      auto& conn = m_conns[i];
      // send count is zero when this connection is receiving only
      if (this->get_send_counts(is_forward)[this->m_peers[i]] == 0)
      {
        continue;
      }
      DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(m_streams[i], m_ev, 0));
      conn->put(
        send_buf + this->get_send_displs_h(is_forward)[conn->get_peer()],
        static_cast<DataType*>(get_peer_addrs(is_forward)[i])
          + get_peer_offsets(is_forward)[i],
        this->get_send_counts(is_forward)[conn->get_peer()] * sizeof(DataType),
        m_streams[i]);
    }
#else
    // Butterfly exchanges for Ray. Worse performance than the above
    // straightforward transfer ordering.
    int my_rank = this->m_loc.get_rank();
    int intra_numa_peer = my_rank ^ 1;
    for (int i = 0; i < num_peers; ++i)
    {
      auto& conn = m_conns[i];
      if (this->get_send_counts(is_forward)[this->m_peers[i]] == 0)
      {
        continue;
      }
      DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(m_streams[i], m_ev, 0));
      // Issues intra-numa copies as they won't conflict or conflicts
      // are still fast enough to ignore
      if (conn->get_peer() == my_rank || conn->get_peer() == intra_numa_peer)
      {
        conn->put(send_buf
                    + this->get_send_displs_h(is_forward)[conn->get_peer()],
                  static_cast<DataType*>(get_peer_addrs(is_forward)[i])
                    + get_peer_offsets(is_forward)[i],
                  this->get_send_counts(is_forward)[conn->get_peer()]
                    * sizeof(DataType),
                  m_streams[i]);
      }
    }
    // Step 1
    {
      int i = ((my_rank % 4) + 2) % 4;
      auto& conn = m_conns[i];
      if (this->get_send_counts(is_forward)[this->m_peers[i]] != 0)
      {
        util::MPIPrintStreamDebug()
          << "Putting to " << conn->get_peer() << "\n";
        conn->put(send_buf
                    + this->get_send_displs_h(is_forward)[conn->get_peer()],
                  static_cast<DataType*>(get_peer_addrs(is_forward)[i])
                    + get_peer_offsets(is_forward)[i],
                  this->get_send_counts(is_forward)[conn->get_peer()]
                    * sizeof(DataType),
                  m_streams[i]);
      }
    }
    // Step 2
    {
      int i = my_rank % 4;
      i = (i - (i % 2) + 2 + ((i % 2) ^ 1)) % 4;
      auto& conn = m_conns[i];
      if (this->get_send_counts(is_forward)[this->m_peers[i]] != 0)
      {
        util::MPIPrintStreamDebug()
          << "Putting to " << conn->get_peer() << "\n";
        conn->put(send_buf
                    + this->get_send_displs_h(is_forward)[conn->get_peer()],
                  static_cast<DataType*>(get_peer_addrs(is_forward)[i])
                    + get_peer_offsets(is_forward)[i],
                  this->get_send_counts(is_forward)[conn->get_peer()]
                    * sizeof(DataType),
                  m_streams[i]);
      }
    }
#endif
    m_p2p.barrier(m_conns, m_streams, num_peers);
    for (int i = 0; i < num_peers; ++i)
    {
      if (this->get_recv_counts(is_forward)[this->m_peers[i]] != 0)
      {
        DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev, m_streams[i]));
        DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(stream, m_ev, 0));
      }
    }
  }

  void release_buf(DataType* buf) override
  {
    // Buffers are reused without releasing
    return;
  }
};

} // namespace tensor
} // namespace distconv
