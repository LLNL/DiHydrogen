#include "p2p/request.hpp"

#include "p2p/connection.hpp"
#include "p2p/logging.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"

using namespace p2p::logging;

namespace p2p
{
Request::Request()
  : Request(Kind::NULL_REQUEST, nullptr, nullptr, 0, 0, nullptr)
{}

Request::Request(Connection* conn, MPI_Request req, cudaStream_t stream)
  : Request(Kind::DEFAULT, conn, req, stream)
{}

Request::Request(Kind kind,
                 Connection* conn,
                 MPI_Request req,
                 cudaStream_t stream)
  : Request(kind, conn, &req, 1, stream, nullptr)
{}

Request::Request(Connection* conn,
                 MPI_Request req1,
                 MPI_Request req2,
                 cudaStream_t stream)
  : Request(Kind::DEFAULT, conn, req1, req2, stream)
{}

Request::Request(Kind kind,
                 Connection* conn,
                 MPI_Request req1,
                 MPI_Request req2,
                 cudaStream_t stream)
  : Request(kind, conn, nullptr, 0, stream, nullptr)
{
  m_requests[0] = req1;
  m_requests[1] = req2;
  m_num_requests = 2;
}

Request::Request(Connection* conn,
                 MPI_Request* mpi_requests,
                 int num_requests,
                 cudaStream_t stream,
                 handler_type handler)
  : Request(Kind::DEFAULT, conn, mpi_requests, num_requests, stream, handler)
{}

Request::Request(Kind kind,
                 Connection* conn,
                 MPI_Request* mpi_requests,
                 int num_requests,
                 cudaStream_t stream,
                 handler_type handler)
  : m_kind(kind),
    m_conn(conn),
    m_num_requests(num_requests),
    m_stream(stream),
    m_handler(handler)
{
  P2P_ASSERT_ALWAYS(m_num_requests <= MAX_MPI_REQUESTS);
  for (int i = 0; i < MAX_MPI_REQUESTS; ++i)
  {
    if (i < m_num_requests)
    {
      m_requests[i] = mpi_requests[i];
    }
    else
    {
      m_requests[i] = MPI_REQUEST_NULL;
    }
  }
}

Request::Request(const Request& req)
  : m_kind(req.m_kind),
    m_conn(req.m_conn),
    m_num_requests(req.m_num_requests),
    m_stream(req.m_stream),
    m_data(req.m_data),
    m_handler(req.m_handler)
{
  for (int i = 0; i < MAX_MPI_REQUESTS; ++i)
  {
    m_requests[i] = req.m_requests[i];
  }
}

Request& Request::operator=(const Request& req)
{
  m_kind = req.m_kind;
  m_conn = req.m_conn;
  m_num_requests = req.m_num_requests;
  m_stream = req.m_stream;
  for (int i = 0; i < MAX_MPI_REQUESTS; ++i)
  {
    m_requests[i] = req.m_requests[i];
  }
  m_data = req.m_data;
  m_handler = req.m_handler;
  return *this;
}

Connection* Request::get_connection()
{
  return m_conn;
}

MPI_Request* Request::get_mpi_requests()
{
  return m_requests;
}

const MPI_Request* Request::get_mpi_requests() const
{
  return m_requests;
}

bool Request::run_post_process(Request* req)
{
  bool completed = true;
  if (m_handler)
  {
    completed = (m_conn->*m_handler)(m_stream, m_data, req);
  }
  else
  {
    int ret = 0;
    switch (m_kind)
    {
    case Kind::CONNECT: ret = m_conn->connect_post(); break;
    case Kind::REGISTER: ret = m_conn->register_addr_post(m_data); break;
    case Kind::NOTIFY: ret = m_conn->notify_post(m_stream); break;
    case Kind::WAIT: ret = m_conn->wait_post(m_stream); break;
    case Kind::NULL_REQUEST:
    case Kind::DEFAULT: break;
    default:
      logging::MPIPrintStreamError() << "Unknown request type\n";
      P2P_ASSERT_ALWAYS(0);
      break;
    }
    completed = ret == 0;
  }
  return completed;
}

int Request::wait_mpi()
{
  if (m_num_requests > 0)
  {
    return m_conn->m_mpi.wait_requests(m_requests, m_num_requests);
  }
  else
  {
    return 0;
  }
}

int Request::process()
{
  wait_mpi();
  Request req;
  if (!run_post_process(&req))
  {
    req.process();
  }
  return 0;
}

int Request::add_mpi_request(MPI_Request req)
{
  if (m_num_requests + 1 <= MAX_MPI_REQUESTS)
  {
    m_requests[m_num_requests] = req;
    ++m_num_requests;
    return 0;
  }
  else
  {
    logging::MPIPrintStreamError()
      << "Can't add MPI_Request as the capacity in Request is full\n";
    return 1;
  }
}

int Request::set_kind(Kind kind)
{
  m_kind = kind;
  return 0;
}

Request::Kind Request::get_kind() const
{
  return m_kind;
}

int Request::wait(const Request* requests, int num_requests, internal::MPI& mpi)
{
  int num_mpi_req = 0;
  for (int i = 0; i < num_requests; ++i)
  {
    num_mpi_req += requests[i].m_num_requests;
  }
  MPI_Request mpi_requests[num_mpi_req];
  int mr_idx = 0;
  for (int i = 0; i < num_requests; ++i)
  {
    for (int j = 0; j < requests[i].m_num_requests; ++j)
    {
      mpi_requests[mr_idx] = requests[i].get_mpi_requests()[j];
      ++mr_idx;
    }
  }
  mpi.wait_requests(mpi_requests, num_mpi_req);
  return 0;
}

void Request::set_data(void* data)
{
  logging::MPIPrintStreamDebug() << "Setting data " << data << "\n";
  m_data = data;
}

void Request::set_handler(handler_type handler)
{
  m_handler = handler;
}

void Request::process(const Request* requests,
                      int num_requests,
                      internal::MPI& mpi)
{
  std::vector<Request> remaining_requests;
  for (int i = 0; i < num_requests; ++i)
  {
    remaining_requests.push_back(requests[i]);
  }
  while (!remaining_requests.empty())
  {
    wait(remaining_requests.data(), remaining_requests.size(), mpi);
    int next_idx = 0;
    for (auto& r : remaining_requests)
    {
      Request req;
      bool completed = r.run_post_process(&req);
      if (!completed)
      {
        remaining_requests[next_idx] = req;
        ++next_idx;
      }
    }
    remaining_requests.resize(next_idx);
  }
}

} // namespace p2p
