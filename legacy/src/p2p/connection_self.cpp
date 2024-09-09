#include "p2p/connection_self.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "p2p/logging.hpp"
#include "p2p/mpi.hpp"

#include <cstring>

using namespace p2p::logging;

namespace p2p {

ConnectionSelf::ConnectionSelf(const internal::MPI &mpi, util::EventPool &ev_pool):
    Connection(mpi.get_rank(), mpi, ev_pool) {
}

ConnectionSelf::~ConnectionSelf() {
  // disconnect must be done synchronously with the peer
  //process. Randomly calling disconnect() can result in deadlocks.
  //disconnect();
}

Request ConnectionSelf::connect_nb() {
  return Request();
}

Request ConnectionSelf::register_addr_nb(void *self, void *peer) {
  add_or_replace_mapped_peer_memory(peer, peer);
  return Request();
}

int ConnectionSelf::send(const void *src, size_t size,
                        cudaStream_t stream) {
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionSelf::recv(void *dst, size_t size,
                         cudaStream_t stream) {
  P2P_ASSERT_ALWAYS(0 && "Not implemented");  
  return 0;
}

int ConnectionSelf::sendrecv(const void *send_src, size_t send_size,
                            void *recv_dst, size_t recv_size,
                            cudaStream_t stream) {
  P2P_ASSERT_ALWAYS(0 && "Not implemented");
  return 0;
}

int ConnectionSelf::put(const void *src, void *dst, size_t size,
                       cudaStream_t stream) {
  if (size == 0) return 0;
  P2P_CHECK_CUDA(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice, stream));
  return 0;
}

int ConnectionSelf::disconnect() {
  return 0;
}

Request ConnectionSelf::notify_nb(cudaStream_t stream) {
  cudaEvent_t e = m_ev_pool.get();
  P2P_CHECK_CUDA(cudaEventRecord(e, stream));
  m_notifications.push_back(std::make_pair(e, stream));
  return Request();
}

Request ConnectionSelf::wait_nb(cudaStream_t stream) {
  auto n = m_notifications.front();
  if (n.second != stream) {
    P2P_CHECK_CUDA(cudaStreamWaitEvent(stream, n.first, 0));
  }
  m_notifications.pop_front();
  return Request();
}

int ConnectionSelf::transfer(void *local_buf, void *peer_buf, size_t size,
                             cudaStream_t stream, bool is_src) {
  if (is_src) {
    return put(local_buf, peer_buf, size, stream);
  } else {
    return 0;
  }
}

} // namespace p2p
