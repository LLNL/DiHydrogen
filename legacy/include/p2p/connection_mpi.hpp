#pragma once

#include <cuda.h>

#include <condition_variable>
#include <list>
#include <thread>

#include "p2p/connection.hpp"
#include "p2p/util_cuda.hpp"
#include <cuda_runtime_api.h>

#define WAIT_USE_MAPPED_MEM

namespace p2p
{

class ConnectionMPI : public Connection
{
public:
  ConnectionMPI(int peer, const internal::MPI& mpi, util::EventPool& ev_pool);
  ~ConnectionMPI() override;
  Request connect_nb() override;
  Request register_addr_nb(void* self, void* peer) override;
  int send(const void* buf, size_t size, cudaStream_t stream) override;
  int recv(void* buf, size_t size, cudaStream_t stream) override;
  int sendrecv(const void* send_buf,
               size_t send_size,
               void* recv_buf,
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

private:
  bool m_use_stream_mem_ops;
  std::thread m_worker;
  std::condition_variable m_cv;
  std::mutex m_mtx;
  cuuint32_t m_req_counter;
  cuuint32_t* m_wait_mem;
  cuuint32_t* m_wait_mem_host;
  enum class ReqType
  {
    SEND,
    RECV,
    SENDRECV
  };
  struct Req
  {
    ReqType m_type;
    void* m_addr1;
    void* m_addr2;
    size_t m_size1;
    size_t m_size2;
    cudaStream_t m_stream;
    cudaEvent_t m_event1;
    cudaEvent_t m_event2;
    cuuint32_t m_wait_val;
    Req(ReqType type,
        void* addr1,
        void* addr2,
        size_t size1,
        size_t size2,
        cudaStream_t stream,
        cudaEvent_t event1,
        cudaEvent_t event2,
        cuuint32_t wait_val)
      : m_type(type),
        m_addr1(addr1),
        m_addr2(addr2),
        m_size1(size1),
        m_size2(size2),
        m_stream(stream),
        m_event1(event1),
        m_event2(event2),
        m_wait_val(wait_val)
    {}
    static Req create_send_request(void* addr,
                                   size_t size,
                                   cudaStream_t stream,
                                   cudaEvent_t event)
    {
      return Req(ReqType::SEND, addr, nullptr, size, 0, stream, event, 0, 0);
    }
    static Req create_recv_request(void* host_addr,
                                   void* dev_addr,
                                   size_t size,
                                   cudaStream_t stream,
                                   cudaEvent_t event,
                                   cuuint32_t wait_val)
    {
      return Req(ReqType::RECV,
                 host_addr,
                 dev_addr,
                 size,
                 0,
                 stream,
                 event,
                 0,
                 wait_val);
    }
    static Req create_sendrecv_request(void* send_host,
                                       size_t send_size,
                                       cudaEvent_t event,
                                       void* recv_host,
                                       size_t recv_size,
                                       cudaEvent_t recv_event,
                                       cuuint32_t wait_val,
                                       cudaStream_t stream)
    {
      return Req(ReqType::SENDRECV,
                 send_host,
                 recv_host,
                 send_size,
                 recv_size,
                 stream,
                 event,
                 recv_event,
                 wait_val);
    }
  };
  std::list<Req> m_requests;
  cudaStream_t m_internal_stream;
  util::PinnedMemoryPool m_pinned_mem_pool;

  static void
  notify_callback(cudaStream_t stream, cudaError_t status, void* user_data);
  static void
  release_host_mem(cudaStream_t stream, cudaError_t status, void* user_data);

  void run_worker();
  void process_requests(std::list<Req>& requests);
  void process_sendrecv(const Req& r);

  int block_stream(cudaStream_t stream, cuuint32_t wait_val);
  int unblock_stream(cuuint32_t wait_val);
  int spin_wait_stream(cudaStream_t stream, cuuint32_t wait_val);
  int unblock_spin_wait(cuuint32_t wait_val);
};

} // namespace p2p
