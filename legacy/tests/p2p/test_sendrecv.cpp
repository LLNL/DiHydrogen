#include "p2p/p2p.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"
#include "p2p/logging.hpp"
#include "test_util.hpp"

#include <iostream>
#include <cassert>

// just connecting with neighbor procs
int test1(int pid, int np, p2p::P2P &p2p) {
  p2p::logging::MPIPrintStreamInfo() << "Testing " << __FUNCTION__ << "\n";  
  int *buf1;
  int len = 1024 * 1024;
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&buf1, sizeof(int) * len));
  //cudaMalloc(&buf2, sizeof(int) * len);
  cudaStream_t st;
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&st));
  
  MPI_Barrier(MPI_COMM_WORLD);  

  p2p::logging::MPIPrintStreamInfo() << "Step 1\n";
  
  int peer = (pid % 2) == 0 ? pid + 1 : pid - 1;
  if (peer >= 0 && peer < np) {
    p2p::P2P::connection_type conn;
    p2p.get_connections(&peer, &conn, 1);
    int *host = new int[len];    
    if ((pid % 2) == 0) {
      for (int i = 0; i < len; ++i) {
        host[i] = i;
      }
      P2P_CHECK_CUDA_ALWAYS(cudaMemcpy(buf1, host, sizeof(int) * len,
                                       cudaMemcpyHostToDevice));
      conn->send(buf1, sizeof(int) * len, st);
      conn->recv(buf1, sizeof(int) * len, st);      
    } else {
      conn->recv(buf1, sizeof(int) * len, st);
      conn->send(buf1, sizeof(int) * len, st);
    }
    P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(st));
    P2P_CHECK_CUDA_ALWAYS(cudaMemcpy(host, buf1, sizeof(int) * len,
                                     cudaMemcpyDeviceToHost));
    for (int i = 0; i < len; ++i) {
      if (host[i] != i){
        p2p::logging::MPIPrintStreamError()
            << "Mismatch at " << i << ": " << host[i] << "\n";
        return 1;
      }
    }
  }


  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << "Step 1 done\n";
  
  p2p::logging::MPIPrintStreamInfo() << "Step 2\n";
  
  peer = (pid % 2) == 0 ? pid - 1 : pid + 1;
  if (peer >= 0 && peer < np) {
    p2p::P2P::connection_type conn;
    p2p.get_connections(&peer, &conn, 1);
    p2p::logging::MPIPrintStreamInfo() << "connected to " << peer << "\n";
    if ((pid % 2) == 1) {
      conn->send(buf1, sizeof(int) * len, st);
      conn->recv(buf1, sizeof(int) * len, st);      
    } else {
      conn->recv(buf1, sizeof(int) * len, st);
      conn->send(buf1, sizeof(int) * len, st);      
    }
  }
  
  P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(st));
  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";
  p2p.disconnect_all();
  return 0;
}

int test2(int pid, int np, p2p::P2P &p2p) {
  p2p::logging::MPIPrintStreamInfo() << "Testing " << __FUNCTION__ << "\n";  
  if (pid < 2) {
    int *buf1;
    int len = 1024 * 1024;
    P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&buf1, sizeof(int) * len));
    cudaStream_t st;
    P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&st));
    int peer = (pid % 2) == 0 ? pid + 1 : pid - 1;
    p2p::P2P::connection_type conn;
    p2p.get_connections(&peer, &conn, 1);    
    if (pid == 0) {
      conn->send(buf1, sizeof(int) * len, st);      
    } else {
      conn->recv(buf1, sizeof(int) * len, st);
#if 0
      while (cudaStreamQuery(st) == cudaErrorNotReady) {
        p2p::logging::MPIPrintStreamDebug()
            << "recv in progress\n";
      }
#endif
    }
    P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(st));
  }
  MPI_Barrier(MPI_COMM_WORLD);
  p2p::logging::MPIPrintStreamInfo() << __FUNCTION__ << " done\n";  
  p2p.disconnect_all();
  return 0;
}

int test_sendrecv(int pid, int np, p2p::P2P &p2p) {
  p2p::logging::MPIPrintStreamInfo() << "Testing " << __FUNCTION__ << "\n";
  int *buf1;
  int len = 1024 * 1024;
  P2P_CHECK_CUDA_ALWAYS(cudaMalloc(&buf1, sizeof(int) * len));
  //cudaMalloc(&buf2, sizeof(int) * len);
  cudaStream_t st;
  P2P_CHECK_CUDA_ALWAYS(cudaStreamCreate(&st));
  
  MPI_Barrier(MPI_COMM_WORLD);  
  
  int peer = (pid % 2) == 0 ? pid + 1 : pid - 1;
  p2p::P2P::connection_type conn;
  p2p.get_connections(&peer, &conn, 1);
  int *host = new int[len];    
  for (int i = 0; i < len; ++i) {
    host[i] = i + pid;
  }
  P2P_CHECK_CUDA_ALWAYS(cudaMemcpy(buf1, host, sizeof(int) * len,
                                   cudaMemcpyHostToDevice));
  conn->sendrecv(buf1, sizeof(int) * len,
                 buf1, sizeof(int) * len,
                 st);
  P2P_CHECK_CUDA_ALWAYS(cudaStreamSynchronize(st));
  P2P_CHECK_CUDA_ALWAYS(cudaMemcpy(host, buf1, sizeof(int) * len,
                                   cudaMemcpyDeviceToHost));
  for (int i = 0; i < len; ++i) {
    if (host[i] != i + peer){
      p2p::logging::MPIPrintStreamError()
          << "Mismatch at " << i << ": host[i]="
          << host[i] << " != " << i + peer << "\n";
      return 1;
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
  
  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  assert(np == 2);

  p2p::P2P p2p(MPI_COMM_WORLD);

  TEST_RUN(test1(pid, np, p2p));
  TEST_RUN(test2(pid, np, p2p));
  TEST_RUN(test_sendrecv(pid, np, p2p));
  
  p2p.disconnect_all();
  p2p::logging::MPIPrintStreamInfo() << "Disconnected\n";
  MPI_Finalize();
  return 0;
}
