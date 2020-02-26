#pragma once

#include "p2p/mpi.hpp"

#include <cuda_runtime.h>

namespace p2p {

class Connection;

class Request {
  constexpr static int MAX_MPI_REQUESTS = 2;
 public:
  using handler_type = bool (Connection::*)(cudaStream_t, void*, Request*);
  
  enum class Kind {CONNECT, REGISTER, NOTIFY, WAIT,
                   DEFAULT, NULL_REQUEST};
  Request();
  Request(Connection *conn,
          MPI_Request req, cudaStream_t stream=0);
  Request(Kind kind, Connection *conn,
          MPI_Request req, cudaStream_t stream=0);
  Request(Connection *conn,
          MPI_Request req1, MPI_Request req2,
          cudaStream_t stream=0);
  Request(Kind kind, Connection *conn,
          MPI_Request req1, MPI_Request req2,
          cudaStream_t stream=0);
  Request(Connection *conn,
          MPI_Request *mpi_requests, int num_requests,
          cudaStream_t stream, handler_type handler);
  Request(Kind kind, Connection *conn,
          MPI_Request *mpi_requests, int num_requests,
          cudaStream_t stream, handler_type handler);

  Request(const Request &req);
  Request &operator=(const Request &req);

  Connection *get_connection();
  const MPI_Request *get_mpi_requests() const;
  MPI_Request *get_mpi_requests();
  int wait_mpi();
  bool run_post_process(Request *req=nullptr);
  int process();
  int add_mpi_request(MPI_Request req);
  int set_kind(Request::Kind kind);
  Kind get_kind() const;
  void set_data(void *data);
  void set_handler(handler_type handler);

  static int wait(const Request *requests, int num_requests,
                  internal::MPI &mpi);

  static void process(const Request *requests, int num_requests,
                      internal::MPI &mpi);
  
 private:
  Kind m_kind;  
  Connection *m_conn;
  MPI_Request m_requests[MAX_MPI_REQUESTS];
  int m_num_requests;
  cudaStream_t m_stream;
  void *m_data = nullptr;
  handler_type m_handler = nullptr;

};

} // namespace p2p
