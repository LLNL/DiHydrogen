#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <thread>

#include "cuda_profiler_api.h"
#include "p2p/logging.hpp"
#include "p2p/p2p.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "test_util.hpp"

__global__ void kernel(long long int sleep_cycles)
{
  auto start = clock64();
  while (true)
  {
    auto now = clock64();
    if (now - start > sleep_cycles)
    {
      break;
    }
  }
  return;
}

int bandwidth_put(int pid,
                  int np,
                  p2p::P2P& p2p,
                  size_t min_size,
                  size_t max_size,
                  int measurement_pid,
                  int iter,
                  bool validate)
{
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << "\n";

  void* bufs[2];
  void* dst_bufs[2];
  cudaStream_t streams[2];

  for (int i = 0; i < 2; ++i)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&bufs[i], max_size));
    P2P_CHECK_CUDA_ALWAYS(cudaMemset(bufs[i], 0, max_size));
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&dst_bufs[i], max_size));
    P2P_CHECK_CUDA_ALWAYS(cudaMemset(dst_bufs[i], 0, max_size));
    P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&streams[i]));
  }

  char* reference = new char[max_size];
  char* host = new char[max_size];
  for (size_t i = 0; i < max_size; ++i)
  {
    reference[i] = pid;
  }
  for (int i = 0; i < 2; ++i)
  {
    P2P_CHECK_CUDA_ALWAYS(
      cudaMemcpy(bufs[i], reference, max_size, cudaMemcpyHostToDevice));
  }

  int rhs = (pid + 1) % np;
  int lhs = (pid + np - 1) % np;
  int peers[2] = {rhs, lhs};

  p2p::P2P::connection_type conns[2];
  p2p.get_connections(peers, conns, 2);
  p2p::logging::MPIPrintStreamInfo() << "Connection established\n";

  void* remote_bufs[2];
  p2p.exchange_addrs(conns, (void**) dst_bufs, remote_bufs, 2);

  cudaEvent_t timing_start, timing_end;
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreate(&timing_start));
  P2P_CHECK_CUDA_ALWAYS(cudaEventCreate(&timing_end));
  cudaEvent_t ev_st_sync;
  P2P_CHECK_CUDA_ALWAYS(
    cudaEventCreateWithFlags(&ev_st_sync, cudaEventDisableTiming));

  int skip = 5;
  for (size_t size = min_size; size <= max_size; size *= 2)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if (pid == 0)
    {
      std::cout << "Testing " << size << "\n";
    }

    P2P_CHECK_CUDA_ALWAYS(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    kernel<<<1, 1, 0, streams[1]>>>((long long int) (1) * 1000 * 1000 * 1000);

    P2P_CHECK_CUDA(cudaEventRecord(ev_st_sync, streams[1]));
    P2P_CHECK_CUDA(cudaStreamWaitEvent(streams[0], ev_st_sync, 0));
    size_t sizes[2] = {size, size};

    for (int i = 0; i < iter + skip; ++i)
    {
      if (i == skip)
      {
        P2P_CHECK_CUDA(cudaEventRecord(timing_start, streams[1]));
      }
      p2p.exchange(conns,
                   (void**) bufs,
                   (void**) dst_bufs,
                   remote_bufs,
                   sizes,
                   sizes,
                   streams,
                   2);
    }

    P2P_CHECK_CUDA(cudaEventRecord(ev_st_sync, streams[0]));
    P2P_CHECK_CUDA(cudaStreamWaitEvent(streams[1], ev_st_sync, 0));

    P2P_CHECK_CUDA(cudaEventRecord(timing_end, streams[1]));
    P2P_CHECK_CUDA_ALWAYS(cudaDeviceSynchronize());

    float elapsed;
    P2P_CHECK_CUDA(cudaEventElapsedTime(&elapsed, timing_start, timing_end));
    float elapsed_in_us = elapsed * 1000 / iter;
    std::stringstream ss;
    ss << pid << " " << size << " " << elapsed_in_us << "\n";
    std::cout << ss.str();
    std::cout << std::flush;

    if (validate)
    {
      p2p::logging::MPIPrintStreamInfo() << "Validating results\n";
      for (int i = 0; i < 2; ++i)
      {
        P2P_CHECK_CUDA_ALWAYS(
          cudaMemcpy(host, dst_bufs[i], size, cudaMemcpyDeviceToHost));
        for (size_t j = 0; j < size; ++j)
        {
          if (host[j] != peers[i])
          {
            p2p::logging::MPIPrintStreamError()
              << "Error detected at (i,j)=(" << i << "," << j << ")\n";
            return 1;
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";
  p2p.disconnect_all();
  return 0;
}

int main(int argc, char* argv[])
{
  size_t min_size = 1;
  size_t max_size = 1 << 24;
  int measurement_pid = 0;
  int iter = 10;
  bool validate = true;

  assert(argc <= 4);

  if (argc == 2)
  {
    min_size = std::atol(argv[1]);
    max_size = min_size;
  }
  if (argc >= 3)
  {
    min_size = std::atol(argv[1]);
    max_size = std::atol(argv[2]);
  }
  if (argc >= 4)
  {
    validate = std::atol(argv[3]) != 0;
  }

  int local_rank = get_local_rank();
  P2P_CHECK_CUDA_ALWAYS(cudaSetDevice(local_rank));

  int mpi_thread_level;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_level);
  switch (mpi_thread_level)
  {
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

  // p2p.enable_nvtx();

  TEST_RUN(bandwidth_put(
    pid, np, p2p, min_size, max_size, measurement_pid, iter, validate));

  p2p.disconnect_all();

  MPI_Finalize();
  return 0;
}
