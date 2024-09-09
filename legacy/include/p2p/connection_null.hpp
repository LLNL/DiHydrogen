#pragma once

#include "p2p/connection.hpp"

namespace p2p
{

class ConnectionNULL : public Connection
{
public:
  ConnectionNULL(int peer, internal::MPI const& mpi, util::EventPool& ev_pool);
  ~ConnectionNULL() override;
  Request connect_nb() override;
  Request register_addr_nb(void* self, void* peer) override;
  int send(void const* src, size_t size, cudaStream_t stream) override;
  int recv(void* dst, size_t size, cudaStream_t stream) override;
  int sendrecv(void const* send_src,
               size_t send_size,
               void* recv_dst,
               size_t recv_size,
               cudaStream_t stream) override;

  int put(void const* src,
          void* dst,
          size_t size,
          cudaStream_t stream) override;

  int transfer(void* local_buf,
               void* peer_buf,
               size_t size,
               cudaStream_t stream,
               bool is_src) override;

  Request notify_nb(cudaStream_t stream) override;
  Request wait_nb(cudaStream_t stream) override;

  int disconnect() override;
};

}  // namespace p2p
