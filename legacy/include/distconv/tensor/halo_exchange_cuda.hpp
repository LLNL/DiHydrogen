#pragma once

#include "distconv/base.hpp"
#include "distconv/tensor/halo_exchange.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv {
namespace tensor {

template <typename DataType, typename AlBackend>
class HaloExchange<DataType, CUDAAllocator, AlBackend> {
 public:
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  using CommType = std::shared_ptr<typename AlBackend::comm_type>;

  HaloExchange(TensorType &tensor): m_tensor(tensor), m_peers(-1) {
    bool exchange_req = false;
    for (int i = 0; i < tensor.get_num_dims(); ++i) {
      exchange_req |= is_exchange_required(i);
    }
    if (exchange_req) {
      // Does not work for shared tensors yet
      assert_always(!tensor.get_distribution().is_shared());
      set_peer_ranks();
    }
  }

  HaloExchange(const HaloExchange<DataType, CUDAAllocator, AlBackend> &x):
      HaloExchange(x.m_tensor) {
    m_peers = x.m_peers;
  }

  HaloExchange &operator=(const HaloExchange &x) {
    m_tensor = x.m_tensor;
    m_peers = x.m_peers;
    m_halo_send.clear();
    m_halo_recv.clear();
    return *this;
  }

  virtual ~HaloExchange() {}

  /*
    rendezvous: synchronize before exchanging halos. Implicitly done
    with MPI. Explicit barrier is used with the P2P-based
    implementation.
   */
  virtual void exchange(const IntVector &widths_rhs_send,
                        const IntVector &widths_rhs_recv,
                        const IntVector &widths_lhs_send,
                        const IntVector &widths_lhs_recv,
                        BoundaryAttributesV<CommType> &comms,
                        cudaStream_t stream_main,
                        bool rendezvous,
                        bool sync_back,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    cudaStream_t prev_streams[2] = {stream_main, stream_main};
    for (int i = 0; i < m_tensor.get_num_dims(); ++i) {
      if (is_exchange_required(i, widths_rhs_send[i], widths_rhs_recv[i],
                               widths_lhs_send[i], widths_lhs_recv[i])) {
        // Synchronize boundary streams
        for (Side side: SIDES) {
          cudaStream_t prev = prev_streams[side];
          cudaStream_t cur_streams[2] = {comms(i, RHS)->get_stream(),
                                         comms(i, LHS)->get_stream()};
          util::wait_stream(prev, cur_streams, 2);
          prev_streams[side] = comms(i, side)->get_stream();
        }
        exchange(i, widths_rhs_send[i], widths_rhs_recv[i],
                 widths_lhs_send[i], widths_lhs_recv[i],
                 comms(i, RHS), comms(i, LHS), rendezvous,
                 is_reverse, skip_unpack, op);
      }
    }
    if (sync_back) {
      for (Side side: SIDES) {
        util::wait_stream(prev_streams[side], stream_main);
      }
    }
  }
  virtual void exchange(BoundaryAttributesV<CommType> &comms,
                        cudaStream_t stream_main,
                        bool rendezvous,
                        bool sync_back,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    exchange(m_tensor.get_halo_width(), m_tensor.get_halo_width(),
             m_tensor.get_halo_width(), m_tensor.get_halo_width(),
             comms, stream_main, rendezvous, sync_back,
             is_reverse, skip_unpack, op);
  }
  virtual void exchange(int dim,
                        int width_rhs_send, int width_rhs_recv,
                        int width_lhs_send, int width_lhs_recv,
                        CommType &comm_rhs,
                        CommType &comm_lhs,
                        bool rendezvous,
                        bool is_reverse,
                        bool skip_unpack,
                        HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) = 0;

  virtual void exchange(int dim,
                        CommType &comm_rhs,
                        CommType &comm_lhs,
                        bool rendezvous, bool is_reverse, bool skip_unpack,
                        HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    int width = m_tensor.get_halo_width(dim);
    exchange(dim, width, width, width, width,
             comm_rhs, comm_lhs, rendezvous, is_reverse,
             skip_unpack, op);
  }

  void unpack(const IntVector &widths_rhs_recv,
              const IntVector &widths_lhs_recv,
              BoundaryAttributesV<cudaStream_t> &streams,
              cudaStream_t stream_main,
              bool sync_back,
              bool is_reverse,
              HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    cudaStream_t prev_streams[2] = {stream_main, stream_main};
    for (int i = 0; i < m_tensor.get_num_dims(); ++i) {
      if (!is_exchange_required(i, 0, widths_rhs_recv[i],
                                0, widths_lhs_recv[i])) {
        continue;
      }
      for (Side side: SIDES) {
        cudaStream_t prev = prev_streams[side];
        cudaStream_t cur_streams[2] = {streams(i, LHS), streams(i, RHS)};
        util::wait_stream(prev, cur_streams, 2);
        prev_streams[side] = streams(i, side);
      }
      unpack(i, widths_rhs_recv[i], widths_lhs_recv[i],
             streams(i, RHS), streams(i, LHS),
             is_reverse, op);
    }
    if (sync_back) {
      for (Side side: SIDES) {
        util::wait_stream(prev_streams[side], stream_main);
      }
    }
  }

  void unpack(BoundaryAttributesV<cudaStream_t> &streams,
              cudaStream_t stream_main,
              bool sync_back,
              bool is_reverse,
              HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    unpack(m_tensor.get_halo_width(), m_tensor.get_halo_width(), streams,
           stream_main, sync_back, is_reverse, op);
  }

  void dump_packed_halo(int dim) {
    int rank = m_tensor.get_locale().get_rank();
    DataType *h = new DataType[get_halo_size(dim)];
    for (auto side: SIDES) {
      if (get_peer(dim, side) == MPI_PROC_NULL) continue;
      std::ofstream out_send;
      cudaMemcpy(h, get_send_buffer(dim, side),
                 get_halo_size(dim) * sizeof(DataType),
                 cudaMemcpyDeviceToHost);
      std::stringstream file_path_send;
      file_path_send << "send_halo_" << rank << "_"
                << dim << "_" << side << ".txt";
      out_send.open(file_path_send.str(), std::ios::out | std::ios::trunc);
      for (size_t i = 0; i < get_halo_size(dim); ++i) {
        out_send << h[i] << std::endl;
      }
      out_send.close();
      cudaMemcpy(h, get_recv_buffer(dim, side),
                 get_halo_size(dim) * sizeof(DataType),
                 cudaMemcpyDeviceToHost);
      std::ofstream out_recv;
      std::stringstream file_path_recv;
      file_path_recv << "recv_halo_" << rank << "_"
                     << dim << "_" << side << ".txt";
      out_recv.open(file_path_recv.str(), std::ios::out | std::ios::trunc);
      for (size_t i = 0; i < get_halo_size(dim); ++i) {
        out_recv << h[i] << std::endl;
      }
      out_recv.close();
    }
    delete[] h;
    return;
  }
  void dump_recv_halo(int dim) {
    int rank = m_tensor.get_locale().get_rank();
    DataType *h = new DataType[get_halo_size(dim)];
    for (auto side: SIDES) {
      cudaMemcpy(h, get_recv_buffer(dim, side),
                 get_halo_size(dim) * sizeof(DataType),
                 cudaMemcpyDeviceToHost);
      std::ofstream out;
      std::stringstream file_path;
      file_path << "recv_halo_" << rank << "_"
                << dim << "_" << side << ".txt";
      out.open(file_path.str(), std::ios::out | std::ios::trunc);
      for (size_t i = 0; i < get_halo_size(dim); ++i) {
        out << h[i] << std::endl;
      }
      out.close();
    }
    delete[] h;
    return;
  }

 protected:
  TensorType &m_tensor;
  BoundaryAttributesV<Memory<CUDAAllocator>> m_halo_send;
  BoundaryAttributesV<Memory<CUDAAllocator>> m_halo_recv;
  BoundaryAttributesV<int> m_peers;

  int &get_peer(int dim, Side side) {
    return m_peers(dim, side);
  }

  virtual size_t get_halo_size(int dim, int width) const {
    auto local_real_shape = m_tensor.get_local_real_shape();
    local_real_shape[dim] = width;
    return local_real_shape.get_size();
  }

  size_t get_halo_size(int dim) const {
    return get_halo_size(dim, m_tensor.get_distribution().get_overlap(dim));
  }

  virtual void *get_send_buffer(int dim, Side side) {
    return m_halo_send(dim, side).get();
  }

  virtual void *get_recv_buffer(int dim, Side side) {
    return m_halo_recv(dim, side).get();
  }

  virtual void ensure_halo_buffers(int dim) {
    size_t s = get_halo_size(dim) * sizeof(DataType);
    assert_always(s > 0);
    for (auto side: SIDES) {
      if (get_peer(dim, side) == MPI_PROC_NULL) continue;
      if (m_halo_send(dim, side).is_null()) {
        m_halo_send(dim, side).allocate(s);
        m_halo_send(dim, side).memset(0, 0);
        // ensure memset is completed
        DISTCONV_CHECK_CUDA(cudaStreamSynchronize(0));
      }
      if (m_halo_recv(dim, side).is_null()) {
        m_halo_recv(dim, side).allocate(s);
        m_halo_recv(dim, side).memset(0, 0);
        // ensure memset is completed
        DISTCONV_CHECK_CUDA(cudaStreamSynchronize(0));
      }
    }
  }

  virtual bool is_exchange_required(int dim,
                                    int width_rhs_send, int width_rhs_recv,
                                    int width_lhs_send, int width_lhs_recv) {
    const auto &dist = m_tensor.get_distribution();
    return dist.is_distributed(dim) &&
        dist.get_split_shape()[dim] > 1 &&
        (width_rhs_send > 0 || width_rhs_recv > 0 ||
         width_lhs_send > 0 || width_lhs_recv > 0) &&
        (m_tensor.get_local_size() > 0);
  }

  bool is_exchange_required(int dim) {
    int halo_width = m_tensor.get_halo_width(dim);
    return is_exchange_required(dim, halo_width, halo_width,
                                halo_width, halo_width);
  }

  int find_peer_rank(int dim, Side side) {
    if (!is_exchange_required(dim)) {
      return MPI_PROC_NULL;
    }

    // Handle empty tensors
    if (m_tensor.get_local_size() == 0) {
      return MPI_PROC_NULL;
    }

    const auto &dist = m_tensor.get_distribution();
    const auto &locale_shape = dist.get_locale_shape();

    int peer_dim_idx = m_tensor.get_proc_index()[dim];
    if (side == Side::RHS) {
      peer_dim_idx += 1;
    } else {
      peer_dim_idx -= 1;
    }

    // processes located at either edge
    if (peer_dim_idx < 0 || peer_dim_idx >= (int)locale_shape[dim]) {
      return MPI_PROC_NULL;
    }

    auto proc_idx = m_tensor.get_proc_index();
    proc_idx[dim] = peer_dim_idx;

    // if the next tensor size is empty, do not send
    if (m_tensor.get_dimension_rank_offset(dim, proc_idx[dim])
        == m_tensor.get_shape()[dim]) {
      return  MPI_PROC_NULL;
    }

    return get_offset(proc_idx, locale_shape);
  }

  void set_peer_ranks() {
    apply_to_sides(m_tensor.get_num_dims(), [&](int dim, Side side) {
        get_peer(dim, side) = find_peer_rank(dim, side);
      });
  }

  void pack_dim(int dim, Side side, int width,
                cudaStream_t stream, void *buf,
                bool is_reverse) {
    pack_or_unpack(dim, side, width, stream, buf, true, is_reverse);
  }

  void unpack_dim(int dim, Side side, int width,
                  cudaStream_t stream,
                  void *buf, bool is_reverse,
                  HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    pack_or_unpack(dim, side, width, stream, buf, false,
                   is_reverse, op);
  }

  void pack_or_unpack(int dim, Side side, int width,
                      cudaStream_t stream, void *buf,
                      bool is_pack, bool is_reverse,
                      HaloExchangeAccumOp op=HaloExchangeAccumOp::ID);

  virtual bool unpack(int dim,
                      int width_rhs_recv,
                      int width_lhs_recv,
                      cudaStream_t stream_rhs,
                      cudaStream_t stream_lhs,
                      bool is_reverse,
                      HaloExchangeAccumOp op=HaloExchangeAccumOp::ID) {
    bool unpack_done = false;

    // unpack the peer halo
    for (auto side: SIDES) {
      if (get_peer(dim, side) == MPI_PROC_NULL) continue;
      const int width_recv = side == Side::RHS
          ? width_rhs_recv : width_lhs_recv;
      if (width_recv == 0) continue;
       const cudaStream_t stream = side == Side::RHS
          ? stream_rhs : stream_lhs;
      auto recv_buf = get_recv_buffer(dim, side);
      unpack_done = true;
      // Pack the local halo.
      unpack_dim(dim, side, width_recv, stream, recv_buf,
                 is_reverse, op);
    }
    return unpack_done;
  }

};

} // namespace tensor
} // namespace distconv
