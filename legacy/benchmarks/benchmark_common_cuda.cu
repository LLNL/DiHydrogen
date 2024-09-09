#include "benchmark_common_cuda.hpp"

__global__ void spin_kernel(long long int spin_cycles)
{
  auto start = clock64();
  while (true)
  {
    auto now = clock64();
    if (now - start > spin_cycles)
    {
      break;
    }
  }
  return;
}

void spin_gpu(int spin_ms, cudaStream_t stream)
{
  // Assuming 1 GHz
  long long int ms = 1000 * 1000;
  spin_kernel<<<1, 1, 0, stream>>>(ms * spin_ms);
  return;
}
