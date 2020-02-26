#pragma once

#include "p2p/connection.hpp"

#include <list>

namespace p2p {

class ConnectionSelf: public Connection {
 public:
  ConnectionSelf(const internal::MPI &mpi, util::EventPool &ev_pool);
  ~ConnectionSelf() override;

  int send(const void *src, size_t size,
           cudaStream_t stream) override;
  int recv(void *dst, size_t size,
           cudaStream_t stream) override;
  int sendrecv(const void *send_src, size_t send_size,
               void *recv_dst, size_t recv_size,
               cudaStream_t stream) override;

  int put(const void *src, void *dst, size_t size,
          cudaStream_t stream) override;

  int transfer(void *local_buf, void *peer_buf, size_t size,
               cudaStream_t stream, bool is_src) override;

  Request connect_nb() override;
  Request register_addr_nb(void *self, void *peer) override;  
  Request notify_nb(cudaStream_t stream) override;
  Request wait_nb(cudaStream_t stream) override;
  
  int disconnect() override;
  
 private:
  std::list<std::pair<cudaEvent_t, cudaStream_t>> m_notifications;
};

} // namespace p2p
