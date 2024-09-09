#pragma once

#include "distconv/tensor/halo_exchange_cuda.hpp"

#include <Al.hpp>

#include "p2p/p2p.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeHybrid : public HaloExchange<DataType, Allocator, AlBackend>
{
  using TensorType =
    typename HaloExchange<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchange<DataType, Allocator, AlBackend>::CommType;

public:
  HaloExchangeHybrid(TensorType& tensor, p2p::P2P& p2p)
    : HaloExchange<DataType, Allocator, AlBackend>(tensor),
      m_p2p(p2p),
      m_halo_peer(nullptr),
      m_p2p_enabled(false),
      m_p2p_conn_established(tensor.get_num_dims(), false)
  {}

  HaloExchangeHybrid(const HaloExchangeHybrid& x)
    : HaloExchange<DataType, Allocator, AlBackend>(x.m_tensor),
      m_p2p(x.m_p2p),
      m_halo_peer(nullptr),
      m_p2p_enabled(false),
      m_p2p_conn_established(this->m_tensor.get_num_dims(), false)
  {}

  HaloExchangeHybrid& operator=(const HaloExchangeHybrid& x) = delete;

  virtual ~HaloExchangeHybrid() { close_addrs(); }

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
      CommType& comm = side == Side::RHS ? comm_rhs : comm_lhs;
      const cudaStream_t stream = comm->get_stream();
      const int width_send =
        side == Side::RHS ? width_rhs_send : width_lhs_send;
      const int width_recv =
        side == Side::RHS ? width_rhs_recv : width_lhs_recv;
      auto send_buf = this->get_send_buffer(dim, side);
      auto recv_buf = this->get_recv_buffer(dim, side);
      size_t send_count = this->get_halo_size(dim, width_send);
      size_t recv_count = this->get_halo_size(dim, width_recv);
      // pack the local halo
      util::MPIPrintStreamDebug()
        << "Packing halo for dimension " << dim << ", " << side;
      this->pack_dim(dim, side, width_send, stream, send_buf, is_reverse);
      // Use put if possible
      if (is_p2p_enabled(dim, side))
      {
        get_conn(dim, side)->put(send_buf,
                                 get_halo_peer(dim, side),
                                 send_count * sizeof(DataType),
                                 stream);
      }
      else
      {
        // Use Al send/recv
        Al::SendRecv<AlBackend, DataType>(static_cast<DataType*>(send_buf),
                                          send_count,
                                          this->get_peer(dim, side),
                                          static_cast<DataType*>(recv_buf),
                                          recv_count,
                                          this->get_peer(dim, side),
                                          *comm);
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
  BoundaryAttributesV<bool> m_p2p_enabled;
  std::vector<bool> m_p2p_conn_established;

  p2p::P2P::connection_type& get_conn(int dim, Side side)
  {
    return m_conns(dim, side);
  }

  p2p::P2P::connection_type* get_conns(int dim) { return m_conns(dim); }

  void*& get_halo_peer(int dim, Side side) { return m_halo_peer(dim, side); }

  void** get_halo_peers(int dim) { return m_halo_peer(dim); }

  bool& is_p2p_enabled(int dim, Side side) { return m_p2p_enabled(dim, side); }

  void ensure_connection(int dim)
  {
    if (!m_p2p_conn_established.at(dim))
    {
      // Connection not created yet
      m_p2p.get_connections(this->m_peers(dim), get_conns(dim), 2);
      // exchange addresses
      void* self_addrs[2] = {nullptr, nullptr};
      for (auto side : SIDES)
      {
        if (get_conn(dim, side))
        {
          self_addrs[side] = this->get_recv_buffer(dim, side);
          is_p2p_enabled(dim, side) = true;
        }
        else
        {
          // Set the conn as NULL so that operations are ignored
          util::MPIPrintStreamDebug()
            << "P2P not possible from rank "
            << this->m_tensor.get_locale().get_rank() << " to rank "
            << this->get_peer(dim, side);
          int null_proc = MPI_PROC_NULL;
          m_p2p.get_connections(&null_proc, &get_conn(dim, side), 1);
          is_p2p_enabled(dim, side) = false;
        }
      }
      util::MPIPrintStreamDebug()
        << "Exchanging local addreess for dimension " << dim << ": "
        << self_addrs[0] << ", " << self_addrs[1];
      m_p2p.exchange_addrs(get_conns(dim), self_addrs, get_halo_peers(dim), 2);
      m_p2p_conn_established.at(dim) = true;
    }
  }

  void close_addrs()
  {
    for (int i = 0; i < this->m_tensor.get_num_dims(); ++i)
    {
      // close_addrs needs to be called even if no connection exists
      // for this rank as other ranks may have. close_addrs calls
      // MPI_Barrier, so all ranks need to join.
      int num_conns = m_p2p_conn_established.at(i) ? 2 : 0;
      // Connection may be used in different places, but the memory
      // registered for the connection in this class must be freed
      // here.
      m_p2p.close_addrs(get_conns(i), get_halo_peers(i), num_conns);
    }
  }
};

}  // namespace tensor
}  // namespace distconv
