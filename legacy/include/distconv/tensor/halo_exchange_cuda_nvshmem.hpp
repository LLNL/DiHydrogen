#pragma once

#include "distconv_config.hpp"

#ifdef DISTCONV_HAS_NVSHMEM

#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/util/nvshmem.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeNVSHMEM : public HaloExchange<DataType, Allocator, AlBackend>
{
public:
  using TensorType =
    typename HaloExchange<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchange<DataType, Allocator, AlBackend>::CommType;

  HaloExchangeNVSHMEM(TensorType& tensor)
    : HaloExchange<DataType, Allocator, AlBackend>(tensor)
  {
    // Preallocates all NVSHMEM buffers as doing that middle of shmem
    // operations seems to cause deadlock
    for (int i = 0; i < this->m_tensor.get_num_dims(); ++i)
    {
      ensure_halo_buffers(i);
      for (auto side : SIDES)
      {
        m_sync(i, side).alloc_buffers();
      }
      if (this->m_tensor.get_split_index()[i] % 2)
      {
        std::swap(m_sync(i, LHS), m_sync(i, RHS));
      }
    }
  }

  virtual ~HaloExchangeNVSHMEM() = default;

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
      return;
    }
    for (auto side : SIDES)
    {
      int const width_send =
        side == Side::RHS ? width_rhs_send : width_lhs_send;
      int const width_recv =
        side == Side::RHS ? width_rhs_recv : width_lhs_recv;
      cudaStream_t stream =
        side == Side::RHS ? comm_rhs->get_stream() : comm_lhs->get_stream();
      exchange(dim,
               side,
               width_send,
               width_recv,
               stream,
               rendezvous,
               is_reverse,
               skip_unpack,
               op);
    }
    return;
  }

protected:
  BoundaryAttributesV<Memory<NVSHMEMAllocator>> m_halo_send_shmem;
  BoundaryAttributesV<Memory<NVSHMEMAllocator>> m_halo_recv_shmem;
  BoundaryAttributesV<util::nvshmem::PairwiseSync> m_sync;

  // Make it visible as it is hidden due to overloading
  using HaloExchange<DataType, Allocator, AlBackend>::get_halo_size;

  virtual size_t get_halo_size(int dim, int width) const
  {
    auto local_real_shape = this->m_tensor.get_max_local_real_shape();
    local_real_shape[dim] = width;
    return local_real_shape.get_size();
  }

  virtual void ensure_halo_buffers(int dim)
  {
    // Note that this is mostly the same as the parent class, however,
    // the receive buffer is a different variable.
    size_t s = this->get_halo_size(dim) * sizeof(DataType);
    if (s == 0)
      return;
    for (auto side : SIDES)
    {
      // SHMEM buffer needs to be symmetric, so the recv buffer must
      // be created by all processes
      if (this->m_halo_send_shmem(dim, side).is_null())
      {
        util::MPIPrintStreamDebug() << "SHMEM size: " << s;
        m_halo_send_shmem(dim, side).allocate(s);
        m_halo_send_shmem(dim, side).memset(0);
        m_halo_recv_shmem(dim, side).allocate(s);
        m_halo_recv_shmem(dim, side).memset(0);
        DISTCONV_CHECK_CUDA(cudaStreamSynchronize(0));
        util::nvshmem::barrier();
      }
    }
  }

  virtual void* get_send_buffer(int dim, Side side) override
  {
    return m_halo_send_shmem(dim, side).get();
  }

  virtual void* get_recv_buffer(int dim, Side side) override
  {
    return m_halo_recv_shmem(dim, side).get();
  }

  virtual void exchange(int dim,
                        Side side,
                        int width_send,
                        int width_recv,
                        cudaStream_t stream,
                        bool rendezvous,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op)
  {
    if (this->get_peer(dim, side) == MPI_PROC_NULL)
      return;
    if (rendezvous)
    {
      m_sync(dim, side).sync(this->get_peer(dim, side),
                             true,
                             true,
                             util::nvshmem::SyncType::FENCE,
                             stream);
    }
    if (width_send > 0)
    {
      auto send_buf = this->get_send_buffer(dim, side);
      auto peer_recv_buf = this->get_recv_buffer(dim, ~side);
      size_t send_count = this->get_halo_size(dim, width_send);
      // pack the local halo
      this->pack_dim(dim, side, width_send, stream, send_buf, is_reverse);
      nvshmemx_putmem_on_stream((void*) peer_recv_buf,
                                send_buf,
                                send_count * sizeof(DataType),
                                this->get_peer(dim, side),
                                stream);
    }
    m_sync(dim, side).sync(this->get_peer(dim, side),
                           true,
                           true,
                           util::nvshmem::SyncType::FENCE,
                           stream);
    if (!skip_unpack && width_recv > 0)
    {
      auto recv_buf = this->get_recv_buffer(dim, side);
      this->unpack_dim(dim, side, width_recv, stream, recv_buf, is_reverse, op);
    }
  }
};

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeNVSHMEMDirect
  : public HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>
{
public:
  using TensorType =
    typename HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::CommType;

  HaloExchangeNVSHMEMDirect(TensorType& tensor)
    : HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>(tensor)
  {}

  virtual ~HaloExchangeNVSHMEMDirect() {}

  using HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::exchange;

  virtual void exchange(int dim,
                        Side side,
                        int width_send,
                        int width_recv,
                        cudaStream_t stream,
                        bool rendezvous,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op) override
  {
    if (this->get_peer(dim, side) == MPI_PROC_NULL)
      return;
    if (rendezvous)
    {
      this->m_sync(dim, side).sync(this->get_peer(dim, side),
                                   true,
                                   true,
                                   util::nvshmem::SyncType::FENCE,
                                   stream);
    }
    if (width_send > 0)
    {
      auto send_buf = this->get_send_buffer(dim, side);
      auto peer_recv_buf = this->get_recv_buffer(dim, ~side);
      // pack the local halo and put them to the remote buffer directly
      this->pack_and_put(dim,
                         side,
                         width_send,
                         stream,
                         send_buf,
                         is_reverse,
                         peer_recv_buf,
                         this->get_peer(dim, side));
    }
    this->m_sync(dim, side).sync(this->get_peer(dim, side),
                                 true,
                                 true,
                                 util::nvshmem::SyncType::FENCE,
                                 stream);
    if (!skip_unpack && width_recv > 0)
    {
      auto recv_buf = this->get_recv_buffer(dim, side);
      this->unpack_dim(dim, side, width_recv, stream, recv_buf, is_reverse, op);
    }
  }

protected:
  virtual void pack_and_put(int dim,
                            Side side,
                            int width,
                            cudaStream_t stream,
                            void* buf,
                            bool is_reverse,
                            void* dst,
                            int peer);
};

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeNVSHMEMFusedNotify
  : public HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>
{
public:
  using TensorType =
    typename HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::TensorType;
  using CommType =
    typename HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::CommType;

  HaloExchangeNVSHMEMFusedNotify(TensorType& tensor)
    : HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>(tensor)
  {}

  virtual ~HaloExchangeNVSHMEMFusedNotify() = default;

  using HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::exchange;

  virtual void exchange(int dim,
                        Side side,
                        int width_send,
                        int width_recv,
                        cudaStream_t stream,
                        bool rendezvous,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op) override
  {
    if (this->get_peer(dim, side) == MPI_PROC_NULL)
      return;

    if (width_send > 0)
    {
      auto send_buf = this->get_send_buffer(dim, side);
      auto peer_recv_buf = this->get_recv_buffer(dim, ~side);
      this->pack_put_notify(dim,
                            side,
                            width_send,
                            stream,
                            send_buf,
                            is_reverse,
                            peer_recv_buf,
                            this->get_peer(dim, side));
    }
    else
    {
      // Need to increment the sync counter. It's done in the pack
      // kernel in the above case.
      this->m_sync(dim, side).inc_counter(stream);
    }

    if (!skip_unpack && width_recv > 0)
    {
      auto recv_buf = this->get_recv_buffer(dim, side);
      this->wait_and_unpack(
        dim, side, width_recv, stream, recv_buf, is_reverse, op);
    }
  }

protected:
  virtual void pack_put_notify(int dim,
                               Side side,
                               int width,
                               cudaStream_t stream,
                               void* buf,
                               bool is_reverse,
                               void* dst,
                               int peer);

  virtual void wait_and_unpack(int dim,
                               Side side,
                               int width,
                               cudaStream_t stream,
                               void* buf,
                               bool is_reverse,
                               HaloExchangeAccumOp op);
};

// Graph-based halo exchange
#ifdef DISTCONV_HAS_CUDA_GRAPH
template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchangeNVSHMEMGraph
  : public HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>
{
public:
  using TensorType =
    typename HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::TensorType;
  HaloExchangeNVSHMEMGraph(TensorType& tensor)
    : HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>(tensor),
      m_graph_ready(false)
  {}
  virtual ~HaloExchangeNVSHMEMGraph() { destroy_graph(); }

  using HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::exchange;

protected:
  BoundaryAttributesV<bool> m_graph_ready;
  BoundaryAttributesV<cudaGraph_t> m_graphs;
  BoundaryAttributesV<cudaGraphExec_t> m_execs;

  void destroy_graph()
  {
    apply_to_sides(this->m_tensor.get_num_dims(), [&](int dim, Side side) {
      if (m_graph_ready(dim, side))
      {
        DISTCONV_CHECK_CUDA(cudaGraphExecDestroy(m_execs(dim, side)));
        DISTCONV_CHECK_CUDA(cudaGraphDestroy(m_graphs(dim, side)));
        m_graph_ready(dim, side) = false;
      }
    });
  }

  virtual void exchange(int dim,
                        Side side,
                        int width_send,
                        int width_recv,
                        cudaStream_t stream,
                        bool rendezvous,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op) override
  {
    if (!m_graph_ready(dim, side))
    {
      DISTCONV_CHECK_CUDA(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
      HaloExchangeNVSHMEM<DataType, Allocator, AlBackend>::exchange(dim,
                                                                    side,
                                                                    width_send,
                                                                    width_recv,
                                                                    stream,
                                                                    rendezvous,
                                                                    is_reverse,
                                                                    skip_unpack,
                                                                    op);
      DISTCONV_CHECK_CUDA(cudaStreamEndCapture(stream, &m_graphs(dim, side)));
      DISTCONV_CHECK_CUDA(cudaGraphInstantiate(
        &m_execs(dim, side), m_graphs(dim, side), nullptr, nullptr, 0));
      m_graph_ready(dim, side) = true;
    }
    DISTCONV_CHECK_CUDA(cudaGraphLaunch(m_execs(dim, side), stream));
  }
};
#endif  // DISTCONV_CUDA_VERSION_MAJOR

}  // namespace tensor
}  // namespace distconv

#endif  // DISTCONV_HAS_NVSHMEM
