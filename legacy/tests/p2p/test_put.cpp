#include <cassert>
#include <iostream>

#include "p2p/logging.hpp"
#include "p2p/p2p.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "test_util.hpp"

// just connecting with neighbor procs
int put_neighbor(int pid, int np, p2p::P2P& p2p, int count, bool wrap_around)
{
  p2p::logging::MPIPrintStreamInfo()
    << __FUNCTION__ << " wrap_around: " << wrap_around << "\n";
  int *send_rhs, *send_lhs;
  int *recv_rhs, *recv_lhs;
  int len = 1024;
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&send_rhs, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(send_rhs, 0, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&send_lhs, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(send_lhs, 0, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&recv_rhs, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(recv_rhs, 0, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&recv_lhs, sizeof(int) * len));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(recv_lhs, 0, sizeof(int) * len));
  cudaStream_t st;
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&st));
  cudaStream_t streams[2] = {st, st};

  MPI_Barrier(MPI_COMM_WORLD);

  p2p::logging::MPIPrintStreamInfo() << "Step 1\n";

  int rhs = pid + 1;
  if (rhs >= np)
  {
    if (wrap_around)
    {
      rhs = 0;
    }
    else
    {
      rhs = MPI_PROC_NULL;
    }
  }
  int lhs = pid - 1;
  if (lhs < 0)
  {
    if (wrap_around)
    {
      lhs = np - 1;
    }
    else
    {
      lhs = MPI_PROC_NULL;
    }
  }

  int peers[2] = {rhs, lhs};
  p2p::P2P::connection_type conns[2];
  p2p.get_connections(peers, conns, 2);
  p2p::logging::MPIPrintStreamInfo() << "Connection established\n";

  void* send_addrs[2] = {send_rhs, send_lhs};
  void* send_addrs_peer[2];
  void* recv_addrs[2] = {recv_rhs, recv_lhs};
  void* recv_addrs_peer[2];

  p2p.exchange_addrs(conns, send_addrs, send_addrs_peer, 2);
  p2p.exchange_addrs(conns, recv_addrs, recv_addrs_peer, 2);

  p2p::logging::MPIPrintStreamInfo() << "Addresses exchanged\n";

  // setup host memory
  int* host = new int[len];
  int* host_lhs = new int[len];
  int* host_rhs = new int[len];
  for (int i = 0; i < len; ++i)
  {
    host[i] = i + pid * len;
    host_lhs[i] = -1;
    host_rhs[i] = -1;
  }
  P2P_CHECK_CUDA_ALWAYS(
    cudaMemcpy(send_rhs, host, sizeof(int) * len, cudaMemcpyHostToDevice));
  P2P_CHECK_CUDA_ALWAYS(
    cudaMemcpy(send_lhs, host, sizeof(int) * len, cudaMemcpyHostToDevice));

  for (int i = 0; i < count; ++i)
  {
    conns[0]->put(send_addrs[0], recv_addrs_peer[0], sizeof(int) * len, st);
    conns[1]->put(send_addrs[1], recv_addrs_peer[1], sizeof(int) * len, st);
    p2p.barrier(conns, streams, 2);
  }

  P2P_CHECK_CUDA_ALWAYS(cudaDeviceSynchronize());
  MPI_Barrier(MPI_COMM_WORLD);

  P2P_CHECK_CUDA_ALWAYS(cudaMemcpyAsync(
    host_rhs, recv_rhs, sizeof(int) * len, cudaMemcpyDeviceToHost, st));
  P2P_CHECK_CUDA_ALWAYS(cudaMemcpyAsync(
    host_lhs, recv_lhs, sizeof(int) * len, cudaMemcpyDeviceToHost, st));

  P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(st));

  for (int i = 0; i < len; ++i)
  {
    if (rhs != MPI_PROC_NULL)
    {
      if (host_rhs[i] != i + len * rhs)
      {
        p2p::logging::MPIPrintStreamInfo()
          << "RHS Mismatch at " << i << "; " << host_rhs[i]
          << " != " << i + len * rhs << "\n";
      }
    }
    if (lhs != MPI_PROC_NULL)
    {
      if (host_lhs[i] != i + len * lhs)
      {
        p2p::logging::MPIPrintStreamInfo()
          << "LHS Mismatch at " << i << "; " << host_lhs[i]
          << " != " << i + len * lhs << "\n";
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  p2p.disconnect_all();

  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";

  return 0;
}

int main(int argc, char* argv[])
{
  int local_rank = get_local_rank();
  std::cerr << "local rank: " << local_rank << "\n";

  P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(local_rank));
  std::cerr << "cuda devise set: " << local_rank << "\n";

  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  p2p::P2P p2p(MPI_COMM_WORLD);

  TEST_RUN(put_neighbor(pid, np, p2p, 1024, false));
  TEST_RUN(put_neighbor(pid, np, p2p, 1024, true));

  p2p.disconnect_all();

  MPI_Finalize();
  return 0;
}
