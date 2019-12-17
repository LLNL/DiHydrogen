#pragma once

#include "distconv/tensor/halo_exchange_cuda.hpp"

#include <Al.hpp>

namespace distconv {
namespace tensor {

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeAL:
      public HaloExchange<DataType, Allocator, AlBackend> {
  using TensorType = typename HaloExchange<
    DataType, Allocator, AlBackend>::TensorType;
  using CommType = typename HaloExchange<
    DataType, Allocator, AlBackend>::CommType;
 public:
  HaloExchangeAL(TensorType &tensor):
      HaloExchange<DataType, Allocator, AlBackend>(tensor) {}
  HaloExchangeAL(const HaloExchangeAL &x):
      HaloExchange<DataType, Allocator, AlBackend>(x) {}

  virtual ~HaloExchangeAL() {}

  using HaloExchange<DataType, Allocator, AlBackend>::exchange;

  void exchange(int dim,
                int width_rhs_send, int width_rhs_recv,
                int width_lhs_send, int width_lhs_recv,
                CommType &comm_rhs,
                CommType &comm_lhs,
                bool rendezvous,
                bool is_reverse,
                bool skip_unpack,
                HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) override {
    if (!this->is_exchange_required(dim, width_rhs_send, width_rhs_recv,
                                    width_lhs_send, width_lhs_recv)) {
      return;
    }

    this->ensure_halo_buffers(dim);

    for (auto side: SIDES) {
      if (this->get_peer(dim, side) == MPI_PROC_NULL) continue;
      CommType &comm = side == Side::RHS ? comm_rhs : comm_lhs;
      const cudaStream_t stream = comm->get_stream();
      const int width_send = side == Side::RHS ? width_rhs_send : width_lhs_send;
      const int width_recv = side == Side::RHS ? width_rhs_recv : width_lhs_recv;
      auto send_buf = this->get_send_buffer(dim, side);
      auto recv_buf = this->get_recv_buffer(dim, side);
      size_t send_count = this->get_halo_size(dim, width_send);
      size_t recv_count = this->get_halo_size(dim, width_recv);
      if (width_send > 0) {
        // pack the local halo
        this->pack_dim(dim, side, width_send, stream, send_buf, is_reverse);
      }
      Al::SendRecv<AlBackend, DataType>(
          static_cast<DataType*>(send_buf), send_count,
          this->get_peer(dim, side),
          static_cast<DataType*>(recv_buf), recv_count,
          this->get_peer(dim, side),
          *comm);
    }
    if (!skip_unpack) {
      this->unpack(dim, width_rhs_recv, width_lhs_recv,
                   comm_rhs->get_stream(), comm_lhs->get_stream(),
                   is_reverse, op);
    }
    return;
  }
};

} // namespace tensor
} // namespace distconv
