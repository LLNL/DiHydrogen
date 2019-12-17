#pragma once

#include "distconv/tensor/shuffle_mpi_cuda.hpp"

#include <Al.hpp>

namespace distconv {
namespace tensor {

template <typename DataType>
class TensorMPICUDAShufflerAL:
      public TensorMPICUDAShuffler<DataType> {
  using TensorType = typename TensorMPICUDAShuffler<DataType>::TensorType;
 public:
  TensorMPICUDAShufflerAL(const TensorType &src_tensor,
                          const TensorType &dst_tensor,
                          Al::MPICUDABackend::comm_type &al_comm,
                          DataType *src_buf=nullptr,
                          DataType *dst_buf=nullptr):
      TensorMPICUDAShuffler<DataType>(src_tensor, dst_tensor, src_buf, dst_buf),
      m_al_comm(al_comm) {
  }

  virtual ~TensorMPICUDAShufflerAL() = default;

 protected:
  Al::MPICUDABackend::comm_type &m_al_comm;

  void transfer(const DataType *send_buf,
                size_t send_buffer_size,
                DataType *recv_buf,
                size_t recv_buffer_size,
                bool is_forward, cudaStream_t stream) override {
    // Assumes stream is the same as m_al_comm.get_stream()
    std::vector<Al::MPICUDABackend::req_type> requests;
    for (int i = 0; i < this->get_num_peers(); ++i) {
      requests.push_back(Al::MPICUDABackend::null_req);
      auto &req = requests.back();
      auto peer = this->m_peers[i];
      Al::NonblockingSendRecv<Al::MPICUDABackend, DataType>(
          send_buf +this->get_send_displs_h(is_forward)[peer],
          this->get_send_counts(is_forward)[peer], peer,
            recv_buf + this->get_recv_displs_h(is_forward)[peer],
          this->get_recv_counts(is_forward)[peer], peer,
          m_al_comm, req);
    }
    for (auto &req: requests) {
      Al::Wait<Al::MPICUDABackend>(req);
    }
  }
};

} // namespace tensor
} // namespace distconv
