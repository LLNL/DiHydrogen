#pragma once

#include <map>

#include "p2p/mpi.hpp"
#include "p2p/request.hpp"
#include "p2p/util_cuda.hpp"
#include <cuda_runtime.h>

namespace p2p
{

class Connection
{
  friend class Request;

public:
  Connection(int peer, internal::MPI const& mpi, util::EventPool& ev_pool);
  virtual ~Connection() = default;
  virtual Request connect_nb() = 0;
  virtual int exchange_addr(void* local_addr, void** remote_addr);
  virtual int register_addr(void* self, void* peer);
  virtual Request register_addr_nb(void* self, void* peer) = 0;
  virtual int deregister_addr(void* mapped_addr);
  virtual int send(void const* buf, size_t size, cudaStream_t stream) = 0;
  virtual int recv(void* buf, size_t size, cudaStream_t stream) = 0;
  virtual int sendrecv(void const* send_buf,
                       size_t send_size,
                       void* recv_buf,
                       size_t recv_size,
                       cudaStream_t stream) = 0;

  virtual int
  put(void const* src, void* dst, size_t size, cudaStream_t stream) = 0;

  virtual int transfer(void* local_buf,
                       void* peer_buf,
                       size_t size,
                       cudaStream_t stream,
                       bool is_src) = 0;

  virtual int notify(cudaStream_t stream);
  virtual int wait(cudaStream_t stream);
  virtual Request notify_nb(cudaStream_t stream) = 0;
  virtual Request wait_nb(cudaStream_t stream) = 0;

  virtual int disconnect() = 0;
  virtual int close_remote_resources() { return 0; }

  int get_dev() const;
  int get_peer() const;

  void* find_mapped_peer_memory(void const* peer);

protected:
  int m_dev;
  int m_peer;
  internal::MPI m_mpi;
  util::EventPool& m_ev_pool;
  bool m_connected;
  // remote address -> local mapped address
  std::map<void const*, void*> m_remote_mem_map;

  virtual int connect_post();
  virtual int register_addr_post(void* data);
  virtual int notify_post(cudaStream_t stream);
  virtual int wait_post(cudaStream_t stream);

  void add_or_replace_mapped_peer_memory(void const* peer, void* mapped_addr);
  void delete_peer_addr(void const* peer);
  void delete_mapped_addr(void const* mapped_addr);
};

}  // namespace p2p
