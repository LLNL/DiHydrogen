#include "p2p/mpi.hpp"
#include "p2p/logging.hpp"
#include "p2p/util.hpp"

//#define P2P_MPI_LOGGING_DEBUG

namespace p2p {
namespace internal {

constexpr int MPI::m_tag;

MPI::MPI(MPI_Comm comm) {
  MPI_Comm new_comm;
  MPI_Comm_dup(comm, &new_comm);
  m_comm = new_comm;
}

void MPI::delete_comm(MPI_Comm *p) {
  if (*p != MPI_COMM_WORLD && *p != MPI_COMM_NULL) {
    MPI_Comm_free(p);
  }
}

int MPI::get_rank() const {
  int rank;
  MPI_Comm_rank(m_comm, &rank);
  return rank;
}

MPI_Comm MPI::get_comm() {
  return m_comm;
}

int MPI::send(const void *buf, size_t size, int dst) {
#ifdef P2P_MPI_LOGGING_DEBUG
  logging::MPIPrintStreamDebug() << "MPI_Send, size: " << size
                                 << ", dst: " << dst
                                 << ", tag: " << MPI::m_tag
                                 << "\n";
#endif
  MPI_Send(buf, size, MPI_BYTE, dst, MPI::m_tag, get_comm());
  return 0;
}

int MPI::isend(const void *buf, size_t size, int dst,
               MPI_Request *req) {
#ifdef P2P_MPI_LOGGING_DEBUG
  logging::MPIPrintStreamDebug() << "MPI_Isend, size: " << size
                                 << ", dst: " << dst
                                 << ", tag: " << MPI::m_tag
                                 << "\n";
#endif
  MPI_Isend(buf, size, MPI_BYTE, dst, MPI::m_tag, get_comm(), req);
  return 0;
}

int MPI::recv(void *buf, size_t size, int src) {
#ifdef P2P_MPI_LOGGING_DEBUG
  logging::MPIPrintStreamDebug() << "MPI_Recv, size: " << size
                                 << ", src: " << src
                                 << ", tag: " << MPI::m_tag
                                 << "\n";
#endif
  MPI_Recv(buf, size, MPI_BYTE, src, MPI::m_tag, get_comm(),
           MPI_STATUS_IGNORE);
  return 0;
}

int MPI::irecv(void *buf, size_t size, int src,
               MPI_Request *req) {
#ifdef P2P_MPI_LOGGING_DEBUG
  logging::MPIPrintStreamDebug() << "MPI_Irecv, size: " << size
                                 << ", src: " << src
                                 << ", tag: " << MPI::m_tag
                                 << "\n";
#endif
  MPI_Irecv(buf, size, MPI_BYTE, src, MPI::m_tag, get_comm(),
            req);
  return 0;
}

int MPI::sendrecv(const void *send_buf, size_t send_size, int dst,
                  void *recv_buf, size_t recv_size, int src) {
  // sendbuf and recvbuf must be disjoint
  P2P_ASSERT_ALWAYS(send_buf != recv_buf);
  MPI_Sendrecv(send_buf, send_size, MPI_BYTE, dst, MPI::m_tag,
               recv_buf, recv_size, MPI_BYTE, dst, MPI::m_tag,
               get_comm(), MPI_STATUS_IGNORE);
  return 0;
}


int MPI::notify(int peer) {
  return send(&m_notification_key, sizeof(m_notification_key), peer);
}

int MPI::inotify(int peer, MPI_Request *req) {
  return isend(&m_notification_key, sizeof(m_notification_key), peer, req);
}

int MPI::wait_notification(int peer) {
  return recv(&m_notification_key, sizeof(m_notification_key), peer);
}

int MPI::iwait_notification(int peer, MPI_Request *req) {
  return irecv(&m_notification_key, sizeof(m_notification_key), peer, req);
}

int MPI::barrier(int peer) {
  int x = 0;
  int y;
  return sendrecv(&x, sizeof(int), peer,
                  &y, sizeof(int), peer);
}

int MPI::barrier() {
  MPI_Barrier(get_comm());
  return 0;
}

int MPI::wait_requests(MPI_Request *requests, int num_requests) {
#ifdef P2P_MPI_LOGGING_DEBUG
  logging::MPIPrintStreamDebug() << "MPI_Waitall with " << num_requests << " requests\n";
#endif
  P2P_CHECK_MPI(MPI_Waitall(num_requests, requests, MPI_STATUS_IGNORE));
  return 0;
}

} // namespace internal
} // namespace p2p
