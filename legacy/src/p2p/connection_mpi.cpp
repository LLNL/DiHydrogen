#include "p2p/connection_mpi.hpp"

#include "p2p/logging.hpp"
#include "p2p/nvtx.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"

namespace p2p
{

ConnectionMPI::ConnectionMPI(int peer,
                             const internal::MPI& mpi,
                             util::EventPool& ev_pool)
  : Connection(peer, mpi, ev_pool), m_req_counter(0)
{
  m_use_stream_mem_ops = util::is_stream_mem_enabled();
  logging::MPIPrintStreamDebug()
    << "Stream memory operations: "
    << (m_use_stream_mem_ops ? "enabled" : "disabled") << "\n";
  if (m_use_stream_mem_ops)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&m_wait_mem, sizeof(cuuint32_t)));
    P2P_CHECK_CUDA_ALWAYS(cudaMemset(m_wait_mem, 0, sizeof(cuuint32_t)));
  }
  else
  {
    P2P_CHECK_CUDA_ALWAYS(
      cudaHostAlloc(&m_wait_mem,
                    sizeof(cuuint32_t),
                    cudaHostAllocMapped | cudaHostAllocWriteCombined));
    *m_wait_mem = 0;
  }
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&m_internal_stream));
  m_connected = true;
  m_worker = std::thread(&ConnectionMPI::run_worker, this);
}

ConnectionMPI::~ConnectionMPI()
{
  disconnect();
}

Request ConnectionMPI::connect_nb()
{
  return Request(
    Request::Kind::CONNECT, this, MPI_REQUEST_NULL, MPI_REQUEST_NULL);
}

Request ConnectionMPI::register_addr_nb(void* self, void* peer)
{
  // This isn't really supported but works as a placeholder.
  add_or_replace_mapped_peer_memory(peer, nullptr);
  return Request();
}

int ConnectionMPI::block_stream(cudaStream_t stream, cuuint32_t wait_val)
{
  if (m_use_stream_mem_ops)
  {
    logging::MPIPrintStreamDebug() << "Block stream " << stream << " with "
                                   << wait_val << " at " << m_wait_mem << "\n";
    P2P_CHECK_CUDA_DRV(cuStreamWaitValue32(
      stream, (CUdeviceptr) m_wait_mem, wait_val, CU_STREAM_WAIT_VALUE_EQ));
    return 0;
  }
  else
  {
    return spin_wait_stream(stream, wait_val);
  }
}

int ConnectionMPI::unblock_stream(cuuint32_t wait_val)
{
  logging::MPIPrintStreamDebug()
    << "Unblock a stream with " << wait_val << " at " << m_wait_mem << "\n";
  if (m_use_stream_mem_ops)
  {
    P2P_CHECK_CUDA_DRV(cuStreamWriteValue32(m_internal_stream,
                                            (CUdeviceptr) m_wait_mem,
                                            wait_val,
                                            CU_STREAM_WRITE_VALUE_DEFAULT));
    return 0;
  }
  else
  {
    return unblock_spin_wait(wait_val);
  }
}

void ConnectionMPI::notify_callback(cudaStream_t stream,
                                    cudaError_t status,
                                    void* user_data)
{
  ConnectionMPI* conn = static_cast<ConnectionMPI*>(user_data);
  P2P_ASSERT_ALWAYS(conn != nullptr);
  conn->m_cv.notify_one();
  return;
}

int ConnectionMPI::send(const void* buf, size_t size, cudaStream_t stream)
{
  logging::MPIPrintStreamDebug() << "Sending msg of size " << size << "\n";
  void* host = m_pinned_mem_pool.get(size);
  P2P_ASSERT_ALWAYS(host != nullptr);
  P2P_CHECK_CUDA(
    cudaMemcpyAsync(host, buf, size, cudaMemcpyDeviceToHost, stream));
  cudaEvent_t ev = m_ev_pool.get();
  P2P_CHECK_CUDA(cudaEventRecord(ev, stream));
  Req r = Req::create_send_request(host, size, stream, ev);
  std::unique_lock<std::mutex> lock(m_mtx);
  m_requests.push_back(r);
  lock.unlock();
  logging::MPIPrintStreamDebug() << "Notifying worker\n";
  m_cv.notify_one();
  return 0;
}

int ConnectionMPI::recv(void* dst, size_t size, cudaStream_t stream)
{
  logging::MPIPrintStreamDebug() << "Receiving msg of size " << size << "\n";
  void* host = m_pinned_mem_pool.get(size);
  cudaEvent_t e = m_ev_pool.get();
  cuuint32_t wait_val = ++m_req_counter;
  block_stream(stream, wait_val);
  P2P_CHECK_CUDA(
    cudaMemcpyAsync(dst, host, size, cudaMemcpyHostToDevice, stream));
  P2P_CHECK_CUDA(cudaEventRecord(e, stream));
  Req r = Req::create_recv_request(host, dst, size, stream, e, wait_val);
  std::unique_lock<std::mutex> lock(m_mtx);
  m_requests.push_back(r);
  lock.unlock();
  logging::MPIPrintStreamDebug() << "Notifying worker\n";
  m_cv.notify_one();
  return 0;
}

int ConnectionMPI::sendrecv(const void* send_buf,
                            size_t send_size,
                            void* recv_buf,
                            size_t recv_size,
                            cudaStream_t stream)
{
  void* host_send = m_pinned_mem_pool.get(send_size);
  void* host_recv = m_pinned_mem_pool.get(recv_size);
  cuuint32_t wait_val = ++m_req_counter;

  P2P_CHECK_CUDA(cudaMemcpyAsync(
    host_send, send_buf, send_size, cudaMemcpyDeviceToHost, stream));
  cudaEvent_t ev = m_ev_pool.get();
  P2P_CHECK_CUDA(cudaEventRecord(ev, stream));

  block_stream(stream, wait_val);

  P2P_CHECK_CUDA(cudaMemcpyAsync(
    recv_buf, host_recv, recv_size, cudaMemcpyHostToDevice, stream));
  cudaEvent_t ev2 = m_ev_pool.get();
  P2P_CHECK_CUDA(cudaEventRecord(ev2, stream));

  Req r = Req::create_sendrecv_request(
    host_send, send_size, ev, host_recv, recv_size, ev2, wait_val, stream);

  std::unique_lock<std::mutex> lock(m_mtx);
  m_requests.push_back(r);
  lock.unlock();

  logging::MPIPrintStreamDebug() << "Notifying worker\n";
  m_cv.notify_one();

  return 0;
}

int ConnectionMPI::put(const void* src,
                       void* dst,
                       size_t size,
                       cudaStream_t stream)
{
  logging::MPIPrintStreamInfo()
    << "Putting to rank " << m_peer << " of size " << size << "\n";
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionMPI::transfer(void* local_buf,
                            void* peer_buf,
                            size_t size,
                            cudaStream_t stream,
                            bool is_src)
{
  if (is_src)
  {
    return send(local_buf, size, stream);
  }
  else
  {
    return recv(local_buf, size, stream);
  }
}

void ConnectionMPI::run_worker()
{
  P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev));
  while (true)
  {
    std::unique_lock<std::mutex> lock(m_mtx);
    if (!m_connected)
    {
      lock.unlock();
      break;
    }
    if (m_requests.size() == 0)
    {
      logging::MPIPrintStreamDebug() << "Worker starts waiting\n";
      m_cv.wait(lock);
      logging::MPIPrintStreamDebug() << "Worker notified\n";
    }
    std::list<Req> requests(m_requests);
    m_requests.clear();
    lock.unlock();
    process_requests(requests);
  }
  logging::MPIPrintStreamDebug() << "No longer connected; worker exiting\n";
  return;
}

void ConnectionMPI::release_host_mem(cudaStream_t stream,
                                     cudaError_t status,
                                     void* user_data)
{
  logging::MPIPrintStreamDebug() << "CB: releasing pooled mem\n";
  std::pair<util::PinnedMemoryPool*, void*>* pair =
    static_cast<std::pair<util::PinnedMemoryPool*, void*>*>(user_data);
  P2P_ASSERT_ALWAYS(pair != nullptr);
  pair->first->release(pair->second);
  delete pair;
  return;
}

void ConnectionMPI::process_requests(std::list<Req>& requests)
{
  for (auto& r : requests)
  {
    if (r.m_type == ReqType::SEND)
    {
      logging::MPIPrintStreamDebug()
        << "Processing SEND (size: " << r.m_size1 << ")\n";
      P2P_CHECK_CUDA_ALWAYS(cudaEventSynchronize(r.m_event1));
      m_ev_pool.release(r.m_event1);
      logging::MPIPrintStreamDebug() << "Transfer to host done\n";
      internal::nvtx_start("MPI_Send");
      m_mpi.send(r.m_addr1, r.m_size1, m_peer);
      internal::nvtx_end();
      m_pinned_mem_pool.release(r.m_addr1);
      logging::MPIPrintStreamDebug() << "Processing SEND done\n";
    }
    else if (r.m_type == ReqType::RECV)
    {
      logging::MPIPrintStreamDebug() << "Processing RECV\n";
      void* host = r.m_addr1;
      internal::nvtx_start("MPI_recv");
      m_mpi.recv(host, r.m_size1, m_peer);
      internal::nvtx_end();
      // Unblocks the user stream
      unblock_stream(r.m_wait_val);
      P2P_CHECK_CUDA_ALWAYS(cudaEventSynchronize(r.m_event1));
      m_ev_pool.release(r.m_event1);
      m_pinned_mem_pool.release(r.m_addr1);
      logging::MPIPrintStreamDebug() << "Processing RECV done\n";
    }
    else if (r.m_type == ReqType::SENDRECV)
    {
      process_sendrecv(r);
    }
    else
    {
      P2P_ASSERT_ALWAYS(0 && "Should not reach here\n");
    }
  }
}

void ConnectionMPI::process_sendrecv(const Req& r)
{
  logging::MPIPrintStreamDebug() << "Processing SENDRECV\n";
  void* send_buf = r.m_addr1;
  size_t send_size = r.m_size1;
  void* recv_buf = r.m_addr2;
  size_t recv_size = r.m_size2;
  // Wait for the transfer of send data to host buffer
  P2P_CHECK_CUDA_ALWAYS(cudaEventSynchronize(r.m_event1));
  m_ev_pool.release(r.m_event1);
  internal::nvtx_start("MPI_sendrecv");
  m_mpi.sendrecv(send_buf, send_size, m_peer, recv_buf, recv_size, m_peer);
  internal::nvtx_end();
  // Unblocks the user stream
  unblock_stream(r.m_wait_val);
  m_pinned_mem_pool.release(send_buf);
  P2P_CHECK_CUDA_ALWAYS(cudaEventSynchronize(r.m_event2));
  m_ev_pool.release(r.m_event2);
  m_pinned_mem_pool.release(recv_buf);
  logging::MPIPrintStreamDebug() << "Processing SENDRECV done\n";
}

int ConnectionMPI::disconnect()
{
  logging::MPIPrintStreamInfo() << "ConnectionMPI::disconnect\n";
  if (!m_connected)
    return 0;

  std::unique_lock<std::mutex> lock(m_mtx);
  m_connected = false;
  lock.unlock();
  logging::MPIPrintStreamDebug() << "Notifying worker to quit\n";
  m_cv.notify_one();
  std::this_thread::yield();
  logging::MPIPrintStreamDebug() << "Joining worker\n";
  m_worker.join();
  logging::MPIPrintStreamDebug() << "Worker joined\n";
  if (m_use_stream_mem_ops)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaFree(m_wait_mem));
  }
  else
  {
    P2P_CHECK_CUDA_ALWAYS(cudaFreeHost(m_wait_mem));
  }
  P2P_CHECK_CUDA_ALWAYS(cudaStreamDestroy(m_internal_stream));
  return 0;
}

Request ConnectionMPI::notify_nb(cudaStream_t stream)
{
  // TODO
  MPI_Request mr = MPI_REQUEST_NULL;
  Request req(Request::Kind::NOTIFY, this, mr, stream);
  return req;
}

Request ConnectionMPI::wait_nb(cudaStream_t stream)
{
  // TODO
  MPI_Request mr = MPI_REQUEST_NULL;
  Request req(Request::Kind::WAIT, this, mr, stream);
  return req;
}

}  // namespace p2p
