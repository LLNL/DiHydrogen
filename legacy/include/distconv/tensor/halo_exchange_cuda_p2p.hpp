#pragma once

#include "distconv/tensor/halo_exchange_cuda.hpp"

#include "p2p/p2p.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeP2P : public HaloExchange<DataType, Allocator, AlBackend>
{
  using TensorType =
    typename HaloExchange<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchange<DataType, Allocator, AlBackend>::CommType;

public:
  HaloExchangeP2P(TensorType& tensor, p2p::P2P& p2p)
    : HaloExchange<DataType, Allocator, AlBackend>(tensor),
      m_p2p(p2p),
      m_halo_peer(nullptr)
  {}

  HaloExchangeP2P(const HaloExchangeP2P& x)
    : HaloExchange<DataType, Allocator, AlBackend>(x.m_tensor),
      m_p2p(x.m_p2p),
      m_halo_peer(nullptr)
  {}

  HaloExchangeP2P& operator=(const HaloExchangeP2P& x) = delete;

  virtual ~HaloExchangeP2P() { close_addrs(); }

  using HaloExchange<DataType, Allocator, AlBackend>::exchange;

  void exchange(int dim,
                int width_rhs_send,
                int width_rhs_recv,
                int width_lhs_send,
                int width_lhs_recv,
                CommType& comm_rhs,
                CommType& comm_lhs,
                bool rendezvous,
                bool is_reverse,
                bool skip_unpack,
                HaloExchangeAccumOp op = HaloExchangeAccumOp::ID) override
  {
    if (!this->is_exchange_required(
          dim, width_rhs_send, width_rhs_recv, width_lhs_send, width_lhs_recv))
    {
      util::MPIPrintStreamDebug()
        << "exchange not required for dimension " << dim;
      return;
    }
    this->ensure_halo_buffers(dim);
    ensure_connection(dim);
    BoundaryAttributes<cudaStream_t> streams(comm_lhs->get_stream(),
                                             comm_rhs->get_stream());
    if (rendezvous)
      m_p2p.barrier(get_conns(dim), streams.data(), 2);
    for (auto side : SIDES)
    {
      if (this->get_peer(dim, side) == MPI_PROC_NULL)
        continue;
      const cudaStream_t stream =
        side == Side::RHS ? comm_rhs->get_stream() : comm_lhs->get_stream();
      const int width_send =
        side == Side::RHS ? width_rhs_send : width_lhs_send;
      auto send_buf = this->get_send_buffer(dim, side);
      if (width_send > 0)
      {
        util::MPIPrintStreamDebug()
          << "Packing halo for dimension " << dim << ", " << side;
        // pack the local halo
        this->pack_dim(dim, side, width_send, stream, send_buf, is_reverse);
        util::MPIPrintStreamDebug() << "Put packed halo";
        // put
        size_t halo_bytes =
          this->get_halo_size(dim, width_send) * sizeof(DataType);
        get_conn(dim, side)->put(
          send_buf, get_halo_peer(dim, side), halo_bytes, stream);
      }
      else
      {
        util::MPIPrintStreamDebug()
          << "nothing to send for dimension " << dim << ", " << side;
      }
    }
    // make sure the remote device waits for the completion of the put
    m_p2p.barrier(get_conns(dim), streams.data(), 2);
    if (!skip_unpack)
    {
      this->unpack(dim,
                   width_rhs_recv,
                   width_lhs_recv,
                   comm_rhs->get_stream(),
                   comm_lhs->get_stream(),
                   is_reverse,
                   op);
    }
    return;
  }

protected:
  p2p::P2P& m_p2p;
  BoundaryAttributesV<p2p::P2P::connection_type> m_conns;
  BoundaryAttributesV<void*> m_halo_peer;

  p2p::P2P::connection_type& get_conn(int dim, Side side)
  {
    return m_conns(dim, side);
  }

  p2p::P2P::connection_type* get_conns(int dim) { return m_conns(dim); }

  void*& get_halo_peer(int dim, Side side) { return m_halo_peer(dim, side); }

  void** get_halo_peers(int dim) { return m_halo_peer(dim); }

  void ensure_connection(int dim)
  {
    if (!get_conn(dim, RHS))
    {
      // Connection not created yet
      m_p2p.get_connections(this->m_peers(dim), get_conns(dim), 2);
      // exchange addresses
      void* self_addrs[2];
      for (auto side : SIDES)
      {
        self_addrs[side] = this->get_recv_buffer(dim, side);
      }
      util::MPIPrintStreamDebug()
        << "Exchanging local addreess for dimension " << dim << ": "
        << self_addrs[0] << ", " << self_addrs[1] << "\n";
      m_p2p.exchange_addrs(get_conns(dim), self_addrs, get_halo_peers(dim), 2);
    }
  }

  void close_addrs()
  {
    for (int i = 0; i < this->m_tensor.get_num_dims(); ++i)
    {
      // Connection may be used in different places, but the memory
      // registered for the connection in this class must be freed
      // here. Note that when RHS connection exists, LHS should also
      // exist.
      if (get_conn(i, RHS))
      {
        m_p2p.close_addrs(get_conns(i), get_halo_peers(i), 2);
      }
    }
  }
};

}  // namespace tensor
}  // namespace distconv
