#include "p2p/connection_mpi.hpp"
#include "p2p/logging.hpp"
#include "p2p/util.hpp"
#include "p2p/util_cuda.hpp"

namespace p2p
{

__global__ void spin_wait_kernel(cuuint32_t v, cuuint32_t volatile* mem)
{
  do
  {
    cuuint32_t cur_val = *mem;
    if (v <= cur_val)
      break;
  } while (true);
}

int ConnectionMPI::spin_wait_stream(cudaStream_t stream, cuuint32_t wait_val)
{
  logging::MPIPrintStreamDebug()
    << "Blocking stream to wait for " << wait_val << "\n";
  spin_wait_kernel<<<1, 1, 0, stream>>>(wait_val, m_wait_mem);
  return 0;
}

int ConnectionMPI::unblock_spin_wait(cuuint32_t wait_val)
{
  logging::MPIPrintStreamDebug()
    << "Unblocking stream waiting for " << wait_val << "\n";
  *m_wait_mem = wait_val;
  return 0;
}

}  // namespace p2p
