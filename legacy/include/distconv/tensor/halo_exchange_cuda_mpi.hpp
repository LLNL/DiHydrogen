#pragma once

#include "distconv/tensor/halo_exchange_cuda.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeMPI : public HaloExchange<DataType, Allocator, AlBackend>
{
  using TensorType =
    typename HaloExchange<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchange<DataType, Allocator, AlBackend>::CommType;

public:
  HaloExchangeMPI(TensorType& tensor)
    : HaloExchange<DataType, Allocator, AlBackend>(tensor)
  {}
  HaloExchangeMPI(HaloExchangeMPI const& x)
    : HaloExchange<DataType, Allocator, AlBackend>(x)
  {}

  virtual ~HaloExchangeMPI() {}

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
    int const tag = 0;

    if (!this->is_exchange_required(
          dim, width_rhs_send, width_rhs_recv, width_lhs_send, width_lhs_recv))
    {
      return;
    }

    MPI_Comm comm = this->m_tensor.get_locale().get_comm();
    MPI_Request send_req[2];
    MPI_Request recv_req[2];
    int num_send_requests = 0;
    int num_recv_requests = 0;
    this->ensure_halo_buffers(dim);

    for (auto side : SIDES)
    {
      if (this->get_peer(dim, side) == MPI_PROC_NULL)
        continue;
      h2::gpu::DeviceStream const stream =
        side == Side::RHS ? comm_rhs->get_stream() : comm_lhs->get_stream();
      int const width_send =
        side == Side::RHS ? width_rhs_send : width_lhs_send;
      int const width_recv =
        side == Side::RHS ? width_rhs_recv : width_lhs_recv;
      auto send_buf = this->get_send_buffer(dim, side);
      auto recv_buf = this->get_recv_buffer(dim, side);
      if (width_recv > 0)
      {
        size_t halo_bytes =
          this->get_halo_size(dim, width_recv) * sizeof(DataType);
        DISTCONV_CHECK_MPI(MPI_Irecv(recv_buf,
                                     halo_bytes,
                                     MPI_BYTE,
                                     this->get_peer(dim, side),
                                     tag,
                                     comm,
                                     &recv_req[num_recv_requests]));
        ++num_recv_requests;
      }
      util::MPIPrintStreamDebug()
        << "Packing halo for dimension " << dim << ", " << side;
      if (width_send > 0)
      {
        // pack the local halo
        this->pack_dim(dim, side, width_send, stream, send_buf, is_reverse);
        util::MPIPrintStreamDebug() << "Sending packed halo";
        // send
        h2::gpu::sync(stream);
        size_t halo_bytes =
          this->get_halo_size(dim, width_send) * sizeof(DataType);
        DISTCONV_CHECK_MPI(MPI_Isend(send_buf,
                                     halo_bytes,
                                     MPI_BYTE,
                                     this->get_peer(dim, side),
                                     tag,
                                     comm,
                                     &send_req[num_send_requests]));
        ++num_send_requests;
      }
    }

    if (num_recv_requests > 0)
    {
      DISTCONV_CHECK_MPI(
        MPI_Waitall(num_recv_requests, recv_req, MPI_STATUS_IGNORE));
    }

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

    if (num_send_requests)
    {
      DISTCONV_CHECK_MPI(
        MPI_Waitall(num_send_requests, send_req, MPI_STATUS_IGNORE));
    }

    return;
  }
};

}  // namespace tensor
}  // namespace distconv
