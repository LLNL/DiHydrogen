#include "p2p/p2p.hpp"
#include "p2p/connection_null.hpp"
#include "p2p/connection_self.hpp"
#include "p2p/connection_ipc.hpp"
#include "p2p/connection_mpi.hpp"
#include "p2p/logging.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"

#include <cstdlib>

using namespace p2p::internal;
using namespace p2p::logging;

namespace p2p {

P2P::P2P(const internal::MPI &mpi): m_mpi(mpi),
                                    m_stream_mem_enabled(false) {
  m_rank = m_mpi.get_rank();
  P2P_CHECK_CUDA_ALWAYS(cudaGetDevice(&m_dev));
  m_stream_mem_enabled = util::is_stream_mem_enabled();
  if (!m_stream_mem_enabled) {
    MPIPrintStreamInfo() << "Stream memory operations not permitted\n";
  }
  if (m_stream_mem_enabled) {
    init_driver_api();
  }

  int name_len;
  MPI_Get_processor_name(m_proc_name, &name_len);
}

P2P::~P2P() {
  disconnect_all();
}

void P2P::enable_nvtx() const {
  cfg.insert_nvtx_mark = true;
}

void P2P::disable_nvtx() const {
  cfg.insert_nvtx_mark = false;
}


int P2P::get_peer_host_names(const int *peers,
                             int num_peers,
                             std::vector<char*> &names) {
  std::vector<MPI_Request> requests;
  names.clear();
  for (int i = 0; i < num_peers; ++i) {
    MPI_Request mr;
    m_mpi.isend(m_proc_name, MPI_MAX_PROCESSOR_NAME,
                peers[i], &mr);
    requests.push_back(mr);
    char *peer_name = new char[MPI_MAX_PROCESSOR_NAME];
    m_mpi.irecv(peer_name, MPI_MAX_PROCESSOR_NAME,
                peers[i], &mr);
    requests.push_back(mr);
    names.push_back(peer_name);
  }
  m_mpi.wait_requests(requests.data(), requests.size());
  return 0;
}

int P2P::get_peer_devices(const int *peers,
                          int num_peers,
                          std::vector<int> &peer_devices) {
  std::vector<MPI_Request> requests;
  peer_devices.resize(num_peers, -1);
  for (int i = 0; i < num_peers; ++i) {
    MPI_Request mr;
    logging::MPIPrintStreamDebug() << "sending device " << m_dev << " to " << peers[i] << "\n";
    m_mpi.isend(&m_dev, sizeof(int), peers[i], &mr);
    requests.push_back(mr);
    logging::MPIPrintStreamDebug() << "receiving device from " << peers[i] << "\n";
    MPI_Request irecv_mr;
    m_mpi.irecv(&peer_devices[i], sizeof(int), peers[i], &irecv_mr);
    requests.push_back(irecv_mr);
  }
  m_mpi.wait_requests(requests.data(), requests.size());
  return 0;
}

int P2P::get_connections(const std::vector<int> &peers,
                         std::vector<connection_type> &conns) {
  conns.resize(peers.size());
  return get_connections(peers.data(), conns.data(), peers.size());
}

int P2P::get_connections(const int *peers,
                         connection_type *conns,
                         int num_peers) {
  std::vector<char*> host_names;
  std::vector<int> devices;
  get_peer_host_names(peers, num_peers, host_names);
  get_peer_devices(peers, num_peers, devices);
  std::vector<Request> requests;
  for (int i = 0; i < num_peers; ++i) {
    int peer = peers[i];
    MPIPrintStreamDebug() << "Getting a connection to " << peer << "\n";
    auto it = m_conn_map.find(peer);
    std::shared_ptr<Connection> conn;
    if (it != m_conn_map.end()) {
      conn = it->second;
    } else {
      conn = connect(peer, host_names[i], devices[i]);
      m_conn_map.insert(std::make_pair(peer, conn));
      if (conn) {
        requests.push_back(conn->connect_nb());
      }
    }
    conns[i] = conn;
  }
  MPIPrintStreamDebug() << requests.size()
                        << " new connections\n";
  Request::process(requests.data(), requests.size(),
                   m_mpi);
  return 0;
}

std::shared_ptr<Connection> P2P::connect(int peer,
                                         char *peer_name,
                                         int peer_dev) {
  if (peer == MPI_PROC_NULL) {
    MPIPrintStreamDebug() << "Creating a null connection\n";
    return std::make_shared<ConnectionNULL>(peer, m_mpi, m_event_pool);
  } else if (peer == m_mpi.get_rank()) {
    MPIPrintStreamDebug() << "Creating a connection to self\n";
    return std::make_shared<ConnectionSelf>(m_mpi, m_event_pool);
  } else if (ConnectionIPC::is_ipc_capable(
      peer, m_mpi, m_proc_name, peer_name, m_dev, peer_dev)) {
    MPIPrintStreamDebug()
        << "Connecting to rank " << peer << " using device "
        << peer_dev << " with IPC\n";
    return std::make_shared<ConnectionIPC>(peer, peer_dev, m_mpi, m_event_pool);
#if 0 // Disables MPI connection
  } else {
    MPIPrintStreamInfo() <<
        "Connecting to rank " << peer << " with MPI\n";
    return std::make_shared<ConnectionMPI>(peer, m_mpi, m_event_pool);
#endif
  }
  return std::shared_ptr<Connection>(nullptr);
}


int P2P::disconnect_all() {
  for (auto &x: m_conn_map) {
    std::shared_ptr<Connection> conn = x.second;
    // it is a nullptr when the connection was not created
    if (conn) {
      conn->disconnect();
      MPIPrintStreamDebug() << "Disconnecting from " << x.first << "\n";
      conn->close_remote_resources();
    }
  }
  // Note that MPI may be already finalized
  int initialized = false;
  MPI_Initialized(&initialized);
  if (initialized) m_mpi.barrier();
  for (auto &x: m_conn_map) {
    std::shared_ptr<Connection> conn = x.second;
    // it is a nullptr when the connection was not created
    if (conn) {
      conn->disconnect();
    }
  }
  m_conn_map.clear();
  return 0;
}

int P2P::disconnect(connection_type *conns, int num_conns) {
  for (int i = 0; i < num_conns; ++i) {
    auto conn = conns[i];
    // it is a nullptr when the connection was not created
    if (conn) {
      int peer = conn->get_peer();
      MPIPrintStreamDebug() << "Disconnecting from " << peer << "\n";
      m_conn_map.erase(peer);
      conn->close_remote_resources();
    }
  }
  m_mpi.barrier();
  for (int i = 0; i < num_conns; ++i) {
    auto conn = conns[i];
    if (conn) {
      conn->disconnect();
    }
  }
  return 0;
}

int P2P::init_driver_api() {
  CUcontext current_ctxt;
  P2P_CHECK_CUDA_DRV_ALWAYS(cuCtxGetCurrent(&current_ctxt));
  CUdevice rt_dev;
  P2P_CHECK_CUDA_DRV_ALWAYS(cuDeviceGet(&rt_dev, m_dev));
  CUcontext rt_ctxt;
  P2P_CHECK_CUDA_DRV_ALWAYS(cuDevicePrimaryCtxRetain(&rt_ctxt, rt_dev));
  if (current_ctxt == nullptr) {
    P2P_CHECK_CUDA_DRV_ALWAYS(cuCtxSetCurrent(rt_ctxt));
  } else {
    CUdevice current_dev;
    P2P_CHECK_CUDA_DRV_ALWAYS(cuCtxGetDevice(&current_dev));
    P2P_ASSERT_ALWAYS(rt_dev == current_dev);
    P2P_ASSERT_ALWAYS(rt_ctxt == current_ctxt);
  }
  return 0;
}

int P2P::barrier(std::vector<std::shared_ptr<Connection>> &connections,
                 std::vector<cudaStream_t> &streams) {
  return barrier(connections.data(), streams.data(), connections.size());
}

int P2P::barrier(std::shared_ptr<Connection> *connections,
                 cudaStream_t *streams,
                 int num_conns) {
  Request *requests = new Request[num_conns*2];
  for (int i = 0; i < num_conns; ++i) {
    requests[i*2] = connections[i]->notify_nb(streams[i]);
    requests[i*2+1] = connections[i]->wait_nb(streams[i]);
  }
  Request::process(requests, num_conns * 2, m_mpi);
  delete[] requests;
  return 0;
}

int P2P::exchange_addrs(std::vector<connection_type> &connections,
                        const std::vector<void*> &local_addrs,
                        std::vector<void*> &peer_addrs) {
  int num_conns = connections.size();
  peer_addrs.resize(num_conns, nullptr);
  return exchange_addrs(connections.data(), local_addrs.data(),
                        peer_addrs.data(), num_conns);
}

int P2P::exchange_addrs(connection_type *connections,
                        void * const *local_addrs,
                        const size_t *local_offsets,
                        void **peer_addrs,
                        size_t *peer_offsets,
                        int num_conns) {
  int num_req_per_conn = 4;
  MPI_Request mpi_requests[num_conns * num_req_per_conn];
  for (int i = 0; i < num_conns; ++i) {
    auto &conn = connections[i];
    m_mpi.isend(&local_addrs[i], sizeof(void*), conn->get_peer(),
                &mpi_requests[i * num_req_per_conn]);
    m_mpi.irecv(&peer_addrs[i], sizeof(void*), conn->get_peer(),
                &mpi_requests[i * num_req_per_conn + 1]);
    if (local_offsets != nullptr) {
      m_mpi.isend(&local_offsets[i], sizeof(size_t), conn->get_peer(),
                  &mpi_requests[i * num_req_per_conn + 2]);
    } else {
      mpi_requests[i * num_req_per_conn + 2] = MPI_REQUEST_NULL;
    }
    if (peer_offsets != nullptr) {
      m_mpi.irecv(&peer_offsets[i], sizeof(size_t), conn->get_peer(),
                  &mpi_requests[i * num_req_per_conn + 3]);
    } else {
      mpi_requests[i * num_req_per_conn + 3] = MPI_REQUEST_NULL;
    }
  }
  m_mpi.wait_requests(mpi_requests, num_conns * num_req_per_conn);
  // register the remote addresses
  Request *requests = new Request[num_conns];
  for (int i = 0; i < num_conns; ++i) {
    requests[i] = connections[i]->register_addr_nb(
        local_addrs[i], peer_addrs[i]);
  }
  Request::wait(requests, num_conns, m_mpi);
  for (int i = 0; i < num_conns; ++i) {
    requests[i].run_post_process();
    if (peer_addrs[i]) {
      peer_addrs[i] = connections[i]->find_mapped_peer_memory(peer_addrs[i]);
    }
  }
  delete[] requests;
  return 0;
}

int P2P::exchange_addrs(connection_type *connections,
                        void * const *local_addrs,
                        void **peer_addrs,
                        int num_conns) {
  return exchange_addrs(connections, local_addrs, nullptr,
                        peer_addrs, nullptr, num_conns);
}

int P2P::close_addrs(connection_type *connections,
                     void **peer_mapped_addrs,
                     int num_conns) {
  std::vector<MPI_Request> requests;
  for (int i = 0; i < num_conns; ++i) {
    int peer = connections[i]->get_peer();
    MPIPrintStreamDebug() << "Closing mapped memory of rank " << peer << "\n";
    if (peer_mapped_addrs[i]) {
      connections[i]->deregister_addr(peer_mapped_addrs[i]);
    }
    MPI_Request notify_req;
    m_mpi.inotify(peer, &notify_req);
    MPI_Request wait_req;
    m_mpi.iwait_notification(peer, &wait_req);
    requests.push_back(notify_req);
    requests.push_back(wait_req);
  }
  m_mpi.wait_requests(requests.data(), requests.size());
  return 0;
}

int P2P::exchange(connection_type *connections,
                  void * const *local_src_bufs,
                  void **local_dst_bufs,
                  void **peer_dst_bufs,
                  size_t *local_sizes,
                  size_t *peer_sizes,
                  cudaStream_t *streams,
                  int num_conns) {
  int num_ipc_conns = 0;
  P2P::connection_type *ipc_conns = new P2P::connection_type[num_conns];
  cudaStream_t *ipc_streams = new cudaStream_t[num_conns];
  for (int i = 0; i < num_conns; ++i) {
    auto conn = connections[i];
    const auto conn_ptr = conn.get();
    if (typeid(*conn_ptr) == typeid(ConnectionMPI)) {
      logging::MPIPrintStreamDebug() << "MPI exchange with " << conn->get_peer() << "\n";
      conn->sendrecv(local_src_bufs[i], local_sizes[i], local_dst_bufs[i], peer_sizes[i], streams[i]);
    } else if (typeid(*conn_ptr) == typeid(ConnectionIPC)) {
      p2p::logging::MPIPrintStreamDebug() << "IPC exchange with " << conn->get_peer() << "\n";
      conn->put(local_src_bufs[i], peer_dst_bufs[i], local_sizes[i], streams[i]);
      ipc_conns[num_ipc_conns] = conn;
      ipc_streams[num_ipc_conns] = streams[i];
      ++num_ipc_conns;
    }
  }

  barrier(ipc_conns, ipc_streams, num_ipc_conns);
  return 0;
}

} // namespace p2p
