#include "p2p/connection_null.hpp"

#include "p2p/util.hpp"

namespace p2p
{

ConnectionNULL::ConnectionNULL(int peer,
                               internal::MPI const& mpi,
                               util::EventPool& ev_pool)
  : Connection(MPI_PROC_NULL, mpi, ev_pool)
{
  P2P_ASSERT_ALWAYS(peer == MPI_PROC_NULL);
}

ConnectionNULL::~ConnectionNULL()
{}

Request ConnectionNULL::connect_nb()
{
  return Request();
}

Request ConnectionNULL::register_addr_nb(void* self, void* peer)
{
  add_or_replace_mapped_peer_memory(peer, nullptr);
  return Request();
}

int ConnectionNULL::send(void const* src, size_t size, cudaStream_t stream)
{
  return 0;
}

int ConnectionNULL::recv(void* dst, size_t size, cudaStream_t stream)
{
  return 0;
}

int ConnectionNULL::sendrecv(void const* send_src,
                             size_t send_size,
                             void* recv_dst,
                             size_t recv_size,
                             cudaStream_t stream)
{
  return 0;
}

int ConnectionNULL::put(void const* src,
                        void* dst,
                        size_t size,
                        cudaStream_t stream)
{
  return 0;
}

int ConnectionNULL::transfer(void* local_buf,
                             void* peer_buf,
                             size_t size,
                             cudaStream_t stream,
                             bool is_src)
{
  return 0;
}

Request ConnectionNULL::notify_nb(cudaStream_t stream)
{
  return Request(Request::Kind::NOTIFY, this, MPI_REQUEST_NULL, stream);
}

Request ConnectionNULL::wait_nb(cudaStream_t stream)
{
  return Request(Request::Kind::WAIT, this, MPI_REQUEST_NULL, stream);
}

int ConnectionNULL::disconnect()
{
  return 0;
}

}  // namespace p2p
