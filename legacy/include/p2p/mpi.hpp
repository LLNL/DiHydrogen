#pragma once

#include "mpi.h"

#include <memory>

namespace p2p
{

namespace internal
{

class MPI
{
public:
  MPI(MPI_Comm comm);
  int get_rank() const;
  int send(const void* buf, size_t size, int dst);
  int isend(const void* buf, size_t size, int dst, MPI_Request* req);
  int recv(void* buf, size_t size, int src);
  int irecv(void* buf, size_t size, int src, MPI_Request* req);
  int sendrecv(const void* send_buf,
               size_t send_size,
               int dst,
               void* recv_buf,
               size_t recv_size,
               int src);

  int notify(int peer);
  int inotify(int peer, MPI_Request* req);
  int wait_notification(int peer);
  int iwait_notification(int peer, MPI_Request* req);
  int barrier(int peer);
  int barrier();
  int wait_requests(MPI_Request* requests, int num_requests);

private:
  MPI_Comm get_comm();
  // std::shared_ptr<MPI_Comm> m_comm;
  MPI_Comm m_comm;
  static void delete_comm(MPI_Comm* p);
  static constexpr int m_tag = 0;
  char m_notification_key = 0;
};

}  // namespace internal
}  // namespace p2p
