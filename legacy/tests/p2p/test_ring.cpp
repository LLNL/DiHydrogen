#include "p2p/p2p.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "p2p/logging.hpp"
#include "p2p/connection_ipc.hpp"
#include "p2p/connection_mpi.hpp"
#include "test_util.hpp"

#include <iostream>
#include <cassert>
#include <typeinfo>

int test_ring(int pid, int np, p2p::P2P &p2p) {
  p2p::logging::MPIPrintStreamInfo() << "Testing " << __FUNCTION__ << "\n";
  int len = 1024 * 1024;
  size_t size = sizeof(int) * len;
  int *bufs[2];
  int *dst_bufs[2];
  cudaStream_t streams[2];  
  for (int i = 0; i < 2; ++i) {
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&bufs[i], size));
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&dst_bufs[i], size));
    P2P_CHECK_CUDA_ALWAYS(cudaMemset(dst_bufs[i], 0, size));
    P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&streams[i]));
  }
  
  MPI_Barrier(MPI_COMM_WORLD);  
  
  int rhs = (pid + 1) % np;
  int lhs = (pid + np - 1) % np;
  int peers[2] = {rhs, lhs};
  p2p::P2P::connection_type conns[2];
  p2p.get_connections(peers, conns, 2);

  void *remote_bufs[2];
  p2p.exchange_addrs(conns, (void **)dst_bufs, remote_bufs, 2);

  // Initialize input
  int *host = new int[len];
  for (int i = 0; i < len; ++i) {
    host[i] = i + pid;
  }
  for (int i = 0; i < 2; ++i) {
    P2P_CHECK_CUDA_ALWAYS(cudaMemcpy(bufs[i], host, size,
                                     cudaMemcpyHostToDevice));
  }

  size_t sizes[2] = {size, size};
  p2p.exchange(conns, (void **)bufs, (void **)dst_bufs, remote_bufs, sizes, sizes, streams, 2);

  // Validation
  for (int i = 0; i < 2; ++i) {
    int peer = i == 0 ? rhs : lhs;
    P2P_CHECK_CUDA_ALWAYS(cudaMemcpyAsync(host, dst_bufs[i], size,
                                          cudaMemcpyDeviceToHost, streams[i]));
    P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(streams[i]));
    for (int j = 0; j < len; ++j) {
      if (host[j] != j + peer){
        p2p::logging::MPIPrintStreamError()
            << "Mismatch at (i, j)=(" << i << ", " << j << "): host[j]="
            << host[j] << " != " << j + peer << "\n";
        return 1;
      }
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";
  p2p.disconnect_all();
  return 0;
}

int main(int argc, char *argv[]) {

  int local_rank = get_local_rank();
  std::cerr << "local rank: " << local_rank << "\n";
  
  P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(local_rank));
  std::cerr << "cuda devise set: " << local_rank << "\n";
  
  int mpi_thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_level);
  switch (mpi_thread_level) {
    case MPI_THREAD_MULTIPLE:
      std::cout << "Supported thread level: MPI_THREAD_MULTIPLE\n";
      break;
    case MPI_THREAD_SINGLE:
      std::cout << "Supported thread level: MPI_THREAD_SINGLE\n";
      break;
    case MPI_THREAD_SERIALIZED:
      std::cout << "Supported thread level: MPI_THREAD_SERIALIZED\n";
      break;
    case MPI_THREAD_FUNNELED:
      std::cout << "Supported thread level: MPI_THREAD_FUNNELED\n";
      break;
  }
  
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  p2p::P2P p2p(MPI_COMM_WORLD);

  TEST_RUN(test_ring(pid, np, p2p));
  
  p2p.disconnect_all();
  p2p::logging::MPIPrintStreamInfo() << "Disconnected\n";
  MPI_Finalize();
  return 0;
}
