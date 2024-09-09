#include "p2p/connection_ipc.hpp"

#include <cstring>

#include "p2p/logging.hpp"
#include "p2p/mpi.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"

using namespace p2p::logging;

namespace p2p
{

bool ConnectionIPC::is_ipc_capable(int peer,
                                   internal::MPI& mpi,
                                   const char* self_name,
                                   const char* peer_name,
                                   int self_dev,
                                   int peer_dev)
{
  logging::MPIPrintStreamDebug()
    << "is IPC capable? self name: " << self_name
    << ", peer name: " << peer_name << ", self device: " << self_dev
    << ", peer device: " << peer_dev << "\n";
  if (std::string(self_name) != std::string(peer_name))
  {
    return false;
  }
  int peer_access;
  P2P_CHECK_CUDA_ALWAYS(
    cudaDeviceCanAccessPeer(&peer_access, self_dev, peer_dev));
  if (peer_access != 0)
  {
    // Peer access possible
    return true;
  }
  // IPC memcpy is still possible if a context can be created
  // at the remote device
  cudaDeviceProp prop;
  P2P_CHECK_CUDA_ALWAYS(cudaGetDeviceProperties(&prop, peer_dev));
  if (prop.computeMode == cudaComputeModeDefault)
  {
    return true;
  }
  return false;
}

ConnectionIPC::ConnectionIPC(int peer,
                             int dev,
                             const internal::MPI& mpi,
                             util::EventPool& ev_pool)
  : Connection(peer, mpi, ev_pool),
    m_dev_peer(dev),
    m_peer_event_opened(false),
    m_peer_enabled(false)
{
  // enable peer access
  enable_peer_access_if_possible();
  // set up event
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreateWithFlags(
    &m_ev, cudaEventInterprocess | cudaEventDisableTiming));
  m_peer_event_opened = false;
  m_connected = false;
}

ConnectionIPC::~ConnectionIPC()
{
  // disconnect must be done synchronously with the peer
  // process. Randomly calling disconnect() can result in deadlocks.
  // disconnect();
}

Request ConnectionIPC::connect_nb()
{
  logging::MPIPrintStreamDebug() << "ConnectIPC::connect_nb\n";
  cudaIpcEventHandle_t ipc_handle_self;
  P2P_CHECK_CUDA_ALWAYS(cudaIpcGetEventHandle(&ipc_handle_self, m_ev));
  MPI_Request isend_req;
  m_mpi.isend(
    &ipc_handle_self, sizeof(cudaIpcEventHandle_t), m_peer, &isend_req);
  MPI_Request irecv_req;
  m_mpi.irecv(
    &m_peer_ev_handle, sizeof(cudaIpcEventHandle_t), m_peer, &irecv_req);
  Request req(Request::Kind::CONNECT, this, isend_req, irecv_req);
  return req;
}

int ConnectionIPC::connect_post()
{
  P2P_CHECK_CUDA_ALWAYS(cudaIpcOpenEventHandle(&m_ev_peer, m_peer_ev_handle));
  m_peer_event_opened = true;
  m_connected = true;
  return 0;
}

void ConnectionIPC::enable_peer_access_if_possible()
{
  if (!m_peer_enabled)
  {
    int peer_access;
    P2P_CHECK_CUDA_ALWAYS(
      cudaDeviceCanAccessPeer(&peer_access, m_dev, m_dev_peer));
    if (peer_access)
    {
      MPIPrintStreamDebug()
        << "Enabling direct access from devices " << m_dev << " to "
        << m_dev_peer << ", current available memory: "
        << (util::get_available_memory() / 1024 / 1024) << " MB\n";
      cudaError_t e = cudaDeviceEnablePeerAccess(m_dev_peer, 0);
      P2P_ASSERT_ALWAYS(e == cudaSuccess
                        || e == cudaErrorPeerAccessAlreadyEnabled);
      m_peer_enabled = true;
      // clear the error status
      cudaGetLastError();
    }
  }
}

Request ConnectionIPC::register_addr_nb(void* self, void* peer)
{
  MPIPrintStreamDebug() << "Registering local addr, " << self
                        << ", and remote addr, " << peer << "\n";
  cudaIpcMemHandle_t self_handle;
  auto* peer_data = new register_addr_data();
  peer_data->first = peer;
  MPI_Request isend_req, irecv_req;
  if (self)
  {
    P2P_CHECK_CUDA(cudaIpcGetMemHandle(&self_handle, self));
    m_mpi.isend(&self_handle, sizeof(cudaIpcMemHandle_t), m_peer, &isend_req);
  }
  else
  {
    isend_req = MPI_REQUEST_NULL;
  }
  if (peer)
  {
    m_mpi.irecv(
      &(peer_data->second), sizeof(cudaIpcMemHandle_t), m_peer, &irecv_req);
  }
  else
  {
    irecv_req = MPI_REQUEST_NULL;
  }

  auto it = m_opened_local_mem.find(self);
  if (it != m_opened_local_mem.end())
  {
    m_opened_local_mem.erase(it);
  }
  m_opened_local_mem.insert(self);

  Request req(Request::Kind::REGISTER, this, isend_req, irecv_req);
  req.set_data(peer_data);
  return req;
}

int ConnectionIPC::register_addr_post(void* data)
{
  P2P_ASSERT_ALWAYS(data != nullptr);
  auto peer_data = static_cast<register_addr_data*>(data);
  P2P_ASSERT_ALWAYS(peer_data != nullptr);
  const void* peer = peer_data->first;
  // peer may be null when there should be no transfer to the peer
  if (!peer)
  {
    return 0;
  }

  void* already_opened = find_mapped_peer_memory(peer);
  if (already_opened)
  {
    MPIPrintStreamDebug() << "Remote memory " << peer << " from device "
                          << m_dev_peer << " already opened at "
                          << already_opened << "\n";
    return 0;
  }

  cudaIpcMemHandle_t peer_handle = peer_data->second;
  if (!m_peer_enabled)
  {
    MPIPrintStreamDebug() << "Changing the current device to the remote device "
                             "to open remote memmory handle\n";
    P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev_peer));
  }

  MPIPrintStreamDebug() << "Opening remote memory, " << peer << ", from device "
                        << m_dev_peer << "\n";
  void* mapped_mem = nullptr;
  P2P_CHECK_CUDA_ALWAYS(cudaIpcOpenMemHandle(
    &mapped_mem, peer_handle, cudaIpcMemLazyEnablePeerAccess));
  P2P_ASSERT_ALWAYS(mapped_mem != nullptr);
  MPIPrintStreamDebug() << "Mapped at " << mapped_mem << "\n";
  if (!m_peer_enabled)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev));
  }
  add_or_replace_mapped_peer_memory(peer, mapped_mem);
  return 0;
}

int ConnectionIPC::deregister_addr(void* mapped_addr)
{
  P2P_ASSERT_ALWAYS(mapped_addr);
  if (!m_peer_enabled)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev_peer));
  }
  MPIPrintStreamDebug() << "Closing remote memory mapped at " << mapped_addr
                        << "\n";
  P2P_CHECK_CUDA_ALWAYS(cudaIpcCloseMemHandle(mapped_addr));
  if (!m_peer_enabled)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev));
  }
  delete_mapped_addr(mapped_addr);
  return 0;
}

int ConnectionIPC::send(const void* src, size_t size, cudaStream_t stream)
{
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionIPC::recv(void* dst, size_t size, cudaStream_t stream)
{
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionIPC::sendrecv(const void* send_src,
                            size_t send_size,
                            void* recv_dst,
                            size_t recv_size,
                            cudaStream_t stream)
{
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionIPC::put(const void* src,
                       void* dst,
                       size_t size,
                       cudaStream_t stream)
{
  logging::MPIPrintStreamDebug()
    << "Put " << size << " bytes from " << src << " on device " << get_dev()
    << " to rank " << get_peer() << " using device " << m_dev_peer
    << " mapped to " << dst << "\n";
  if (size == 0)
    return 0;
  P2P_CHECK_CUDA(
    cudaMemcpyPeerAsync(dst, m_dev_peer, src, m_dev, size, stream));
  return 0;
}

int ConnectionIPC::transfer(void* local_buf,
                            void* peer_buf,
                            size_t size,
                            cudaStream_t stream,
                            bool is_src)
{
  if (is_src)
  {
    return put(local_buf, peer_buf, size, stream);
  }
  else
  {
    return 0;
  }
}

int ConnectionIPC::close_remote_resources()
{
  logging::MPIPrintStreamDebug() << "IPC: Closing remote resources\n";
  for (auto& x : m_remote_mem_map)
  {
    logging::MPIPrintStreamDebug()
      << "Remote memory of " << x.second << " mapped to " << x.first << "\n";
    P2P_ASSERT_ALWAYS(x.first);
    P2P_ASSERT_ALWAYS(x.second);
    if (x.second != nullptr)
    {
      if (!m_peer_enabled)
      {
        P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev_peer));
      }
      logging::MPIPrintStreamDebug()
        << "Closing remote memory handle: " << x.first << "\n";
      P2P_CHECK_CUDA_ALWAYS(cudaIpcCloseMemHandle(x.second));
      if (!m_peer_enabled)
      {
        P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(m_dev));
      }
    }
  }
  m_remote_mem_map.clear();
  if (m_peer_event_opened)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaEventDestroy(m_ev_peer));
    m_peer_event_opened = false;
  }
  return 0;
}

int ConnectionIPC::disconnect()
{
  if (!m_connected)
    return 0;
  P2P_CHECK_CUDA_ALWAYS(cudaEventDestroy(m_ev));
  m_connected = false;
  return 0;
}

Request ConnectionIPC::notify_nb(cudaStream_t stream)
{
  P2P_CHECK_CUDA(cudaEventRecord(m_ev, stream));
  MPI_Request mpi_req;
  m_mpi.inotify(m_peer, &mpi_req);
  Request req(
    this,
    &mpi_req,
    1,
    stream,
    static_cast<Request::handler_type>(&ConnectionIPC::notify_handler));
  return req;
}

bool ConnectionIPC::notify_handler(cudaStream_t stream,
                                   void* data,
                                   Request* req)
{
  MPI_Request mr;
  m_mpi.iwait_notification(m_peer, &mr);
  *req = Request(this, mr);
  return false;
}

Request ConnectionIPC::wait_nb(cudaStream_t stream)
{
  MPI_Request mr;
  m_mpi.iwait_notification(m_peer, &mr);
  Request req(this,
              &mr,
              1,
              stream,
              static_cast<Request::handler_type>(&ConnectionIPC::wait_handler));
  return req;
}

bool ConnectionIPC::wait_handler(cudaStream_t stream, void* data, Request* req)
{
  P2P_CHECK_CUDA(cudaStreamWaitEvent(stream, m_ev_peer, 0));
  MPI_Request mr;
  m_mpi.inotify(m_peer, &mr);
  *req = Request(this, mr);
  return false;
}

}  // namespace p2p
