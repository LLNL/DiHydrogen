#pragma once

#include "distconv_config.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "p2p/config.hpp"
#include "p2p/connection.hpp"
#include "p2p/mpi.hpp"
#include "p2p/util_cuda.hpp"
#include <cuda_runtime.h>

namespace p2p
{

class P2P
{
public:
  using connection_type = std::shared_ptr<Connection>;

  P2P(const internal::MPI& mpi);
  virtual ~P2P();

  void enable_nvtx() const;
  void disable_nvtx() const;

  int get_connections(const int* peers, connection_type* conns, int num_peers);
  int get_connections(const std::vector<int>& peers,
                      std::vector<connection_type>& conns);
  int disconnect_all();
  int disconnect(connection_type* conns, int num_conns);

  int barrier(std::vector<connection_type>& connections,
              std::vector<cudaStream_t>& streams);
  int barrier(std::shared_ptr<Connection>* connections,
              cudaStream_t* streams,
              int num_conns);

  int exchange_addrs(std::vector<connection_type>& connections,
                     const std::vector<void*>& local_addrs,
                     std::vector<void*>& peer_addrs);
  int exchange_addrs(connection_type* connections,
                     void* const* local_addrs,
                     void** peer_addrs,
                     int num_conns);
  int exchange_addrs(connection_type* connections,
                     void* const* local_addrs,
                     const size_t* local_offsets,
                     void** peer_addrs,
                     size_t* peer_offsets,
                     int num_conns);

  int close_addrs(connection_type* connections,
                  void** peer_mapped_addrs,
                  int num_conns);

  /**
   * Perform pair-wise exchange on connections.
   * @param connections Connections to perform exchange
   * @param local_src_bufs Local source buffers
   * @param local_dst_bufs Local destination buffers
   * @param peer_dst_bufs Destinationa buffers at peer processes
   * @param local_sizes Size of local buffers
   * @param peer_sizes Size of peer buffers
   * @param streams Streams for each connection on which exchange is performed
   * @param num_conns Number of connections
   */
  int exchange(connection_type* connections,
               void* const* local_src_bufs,
               void** local_dst_bufs,
               void** peer_dst_bufs,
               size_t* local_sizes,
               size_t* peer_sizes,
               cudaStream_t* streams,
               int num_conns);

private:
  int m_rank;
  int m_dev;
  internal::MPI m_mpi;
  std::map<int, std::shared_ptr<Connection>> m_conn_map;
  bool m_stream_mem_enabled;
  char m_proc_name[MPI_MAX_PROCESSOR_NAME];
  util::EventPool m_event_pool;

  std::shared_ptr<Connection> connect(int peer, char* peer_name, int peer_dev);
  int init_driver_api();

  int get_peer_host_names(const int* peers,
                          int num_peers,
                          std::vector<char*>& names);
  int get_peer_devices(const int* peers,
                       int num_peers,
                       std::vector<int>& peer_devices);
};

} // namespace p2p
