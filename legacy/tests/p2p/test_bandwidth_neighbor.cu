#include "p2p/p2p.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "p2p/logging.hpp"
#include "test_util.hpp"

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <sstream>
#include <thread>
#include <chrono>
#include "cuda_profiler_api.h"

__global__ void kernel(long long int sleep_cycles) {
  auto start = clock64();
  while (true) {
    auto now = clock64();
    if (now - start > sleep_cycles) {
      break;
    }
  }
  return;
}

#define USE_SEPARATE_STREAMS

int bandwidth_put(int pid, int np, p2p::P2P &p2p,
                  size_t min_size, size_t max_size,
                  bool wrap_around,
                  int measurement_pid) {
  p2p::logging::MPIPrintStreamInfo()
      << __FUNCTION__ << "(wrap_around: " << wrap_around
      << ")\n";

  void *send_rhs, *send_lhs;
  void *recv_rhs, *recv_lhs;  
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&send_rhs, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(send_rhs, 0, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&send_lhs, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(send_lhs, 0, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&recv_rhs, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(recv_rhs, 0, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&recv_lhs, max_size));
  P2P_CHECK_CUDA_ALWAYS(cudaMemset(recv_lhs, 0, max_size));

  cudaStream_t st;
  int stream_min_priority;
  P2P_CHECK_CUDA_ALWAYS(cudaDeviceGetStreamPriorityRange(
      &stream_min_priority, nullptr));
  //P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&st));
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreateWithPriority(
      &st, cudaStreamDefault, stream_min_priority));
#ifdef USE_SEPARATE_STREAMS  
  cudaStream_t st2;
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreateWithPriority(
      &st2, cudaStreamDefault, stream_min_priority));
#else
  cudaStream_t st2 = st;  
#endif
  cudaStream_t streams[2] = {st, st2};
  
  int rhs = pid + 1;
  if (rhs >= np) {
    if (wrap_around) {
      rhs = 0;
    } else {
      rhs = MPI_PROC_NULL;
    }
  }
  int lhs = pid - 1;
  if (lhs < 0) {
    if (wrap_around) {
      lhs = np - 1;
    } else {
      lhs = MPI_PROC_NULL;
    }
  }

  p2p::P2P::connection_type conns[2];
  int peers[2] = {rhs, lhs};
  p2p.get_connections(peers, conns, 2);
  p2p::logging::MPIPrintStreamInfo() << "Connection established\n";
  
  auto &conn_rhs = conns[0];
  auto &conn_lhs = conns[1];

  void *send_addrs[2] = {send_rhs, send_lhs};
  void *send_addrs_peer[2];
  void *recv_addrs[2] = {recv_rhs, recv_lhs};
  void *recv_addrs_peer[2];

  p2p.exchange_addrs(conns, send_addrs, send_addrs_peer, 2);
  p2p.exchange_addrs(conns, recv_addrs, recv_addrs_peer, 2);  
  
  cudaEvent_t timing_start, timing_end;
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreate(&timing_start));
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreate(&timing_end));
  cudaEvent_t ev_st_sync;
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreateWithFlags(
      &ev_st_sync, cudaEventDisableTiming));
  
  //int skip = 0;
  for (size_t size = min_size; size <= max_size;
       size *= 2) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (pid == 0) {
      std::cout << "Testing " << size << "\n";
    }
    MPI_Barrier(MPI_COMM_WORLD);    
    P2P_CHECK_CUDA_ALWAYS(cudaDeviceSynchronize());

    if (pid == measurement_pid) {
      kernel<<<1, 1, 0, streams[1]>>>(
          (long long int)(1) * 1000 * 1000 * 1000);
    }
#ifdef USE_SEPARATE_STREAMS
    P2P_CHECK_CUDA(cudaEventRecord(ev_st_sync, streams[1]));
    P2P_CHECK_CUDA(cudaStreamWaitEvent(streams[0], ev_st_sync, 0));
#endif

    P2P_CHECK_CUDA(cudaEventRecord(timing_start, streams[1]));
    conn_rhs->put(send_rhs, recv_addrs_peer[0], size, streams[0]);    
    conn_lhs->put(send_lhs, recv_addrs_peer[1], size, streams[1]);    
    p2p.barrier(conns, streams, 2);
#ifdef USE_SEPARATE_STREAMS
    P2P_CHECK_CUDA(cudaEventRecord(ev_st_sync, streams[0]));
    P2P_CHECK_CUDA(cudaStreamWaitEvent(streams[1], ev_st_sync, 0));
#endif
    P2P_CHECK_CUDA(cudaEventRecord(timing_end, streams[1]));    
    P2P_CHECK_CUDA_ALWAYS(cudaDeviceSynchronize());

    float elaplsed;
    P2P_CHECK_CUDA(cudaEventElapsedTime(&elaplsed,
                                        timing_start, timing_end));
    if (measurement_pid == pid) {
      std::stringstream ss;
      ss << pid << " " << size << " "
         << elaplsed * 1000 << "\n";
      std::cout << ss.str();
      std::cout << std::flush;
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";  
  p2p.disconnect_all();
  return 0;
}


int main(int argc, char *argv[]) {
  size_t min_size = 1;
  size_t max_size = 1 << 24;
  int measurement_pid = 0;

  assert(argc <= 4);

  if (argc == 2) {
    min_size = std::atol(argv[1]);
    max_size = min_size;
  }
  if (argc >= 3) {
    min_size = std::atol(argv[1]);    
    max_size = std::atol(argv[2]);
  }
  if (argc >= 4) {
    measurement_pid = std::atol(argv[3]);    
  }
  
  int local_rank = get_local_rank();
  P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(local_rank));
  
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
  
  p2p::logging::MPIPrintStreamInfo()
      << "CUDA devise set: " << local_rank << "\n";
  
  p2p::P2P p2p(MPI_COMM_WORLD);

  //p2p.enable_nvtx();
  
  TEST_RUN(bandwidth_put(pid, np, p2p, min_size, max_size, false, measurement_pid));
  TEST_RUN(bandwidth_put(pid, np, p2p, min_size, max_size, true, measurement_pid));
  
  p2p.disconnect_all();

  MPI_Finalize();
  return 0;
}
