#pragma once

#include <map>
#include <set>

#include "p2p/connection.hpp"

namespace p2p
{

class ConnectionIPC : public Connection
{
public:
  ConnectionIPC(int peer,
                int dev,
                const internal::MPI& mpi,
                util::EventPool& ev_pool);
  ~ConnectionIPC() override;

  static bool is_ipc_capable(int peer,
                             internal::MPI& mpi,
                             const char* self_name,
                             const char* peer_name,
                             int self_dev,
                             int peer_dev);

  Request connect_nb() override;
  int connect_post() override;

  Request register_addr_nb(void* self, void* peer) override;
  int deregister_addr(void* mapped_addr) override;

  int send(const void* src, size_t size, cudaStream_t stream) override;
  int recv(void* dst, size_t size, cudaStream_t stream) override;
  int sendrecv(const void* send_src,
               size_t send_size,
               void* recv_dst,
               size_t recv_size,
               cudaStream_t stream) override;

  int put(const void* src,
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
  int close_remote_resources() override;

private:
  int m_dev_peer;
  cudaEvent_t m_ev;
  cudaIpcEventHandle_t m_peer_ev_handle;
  cudaEvent_t m_ev_peer;
  bool m_peer_event_opened;
  bool m_peer_enabled;
  std::set<void*> m_opened_local_mem;

  void enable_peer_access_if_possible();
  int register_peer_memory(const void* peer, cudaIpcMemHandle_t peer_handle);

  using register_addr_data = std::pair<const void*, cudaIpcMemHandle_t>;
  int register_addr_post(void* data) override;

  bool notify_handler(cudaStream_t stream, void* data, Request* req);
  bool wait_handler(cudaStream_t stream, void* data, Request* req);
};

}  // namespace p2p
