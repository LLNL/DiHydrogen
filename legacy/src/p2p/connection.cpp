#include "p2p/connection.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "p2p/logging.hpp"

using namespace p2p::logging;

namespace p2p {

Connection::Connection(int peer, const internal::MPI &mpi,
                       util::EventPool &ev_pool):
    m_peer(peer), m_mpi(mpi), m_ev_pool(ev_pool),
    m_connected(false) {
  // set remote device ID
  P2P_CHECK_CUDA_ALWAYS(cudaGetDevice(&m_dev));
  MPIPrintStreamDebug() << "Using device " << m_dev << "\n";
}

int Connection::connect_post() {
  return 0;
}

int Connection::register_addr(void *self, void *peer) {
  auto r = register_addr_nb(self, peer);
  Request::wait(&r, 1, m_mpi);
  r.run_post_process();
  return 0;
}

int Connection::register_addr_post(void *data) {
  return 0;
}

int Connection::deregister_addr(void *mapped_addr) {
  return 0;
}

int Connection::exchange_addr(void *local_addr,
                              void **remote_addr) {
  m_mpi.sendrecv(&local_addr, sizeof(void*), m_peer,
                 remote_addr, sizeof(void*), m_peer);
  register_addr(local_addr, *remote_addr);
  if (*remote_addr) {
    *remote_addr = find_mapped_peer_memory(*remote_addr);
  }
  return 0;
}

int Connection::notify(cudaStream_t stream) {
  auto req = notify_nb(stream);
  return req.process();
}

int Connection::wait(cudaStream_t stream) {
  auto req = wait_nb(stream);
  return req.process();
}

int Connection::notify_post(cudaStream_t stream) {
  return 0;
}

int Connection::wait_post(cudaStream_t stream) {
  return 0;
}

int Connection::get_peer() const {
  return m_peer;
}

int Connection::get_dev() const {
  return m_dev;
}

void Connection::add_or_replace_mapped_peer_memory(const void *peer,
                                                   void *mapped_addr) {
  auto it = m_remote_mem_map.find(peer);
  if (it != m_remote_mem_map.end()) {
    m_remote_mem_map.erase(it);
  }
  m_remote_mem_map.insert(std::make_pair(peer, mapped_addr));
  return;
}

void *Connection::find_mapped_peer_memory(const void *peer) {
  auto it = m_remote_mem_map.find(peer);
  if (it == m_remote_mem_map.end()) return nullptr;
  return it->second;
}

void Connection::delete_peer_addr(const void *peer) {
  auto it = m_remote_mem_map.find(peer);
  P2P_ASSERT_ALWAYS(it != m_remote_mem_map.end() &&
                    "Peer memory not mapped\n");
  m_remote_mem_map.erase(it);
  return;
}

void Connection::delete_mapped_addr(const void *mapped_addr) {
  bool matched = false;
  for (auto it = m_remote_mem_map.begin(); it != m_remote_mem_map.end();
       ++it) {
    if (it->second == mapped_addr) {
      m_remote_mem_map.erase(it);
      matched = true;
      break;
    }
  }
  P2P_ASSERT_ALWAYS(matched && "Mapped addr not found\n");
  return;
}

} // namespace p2p
