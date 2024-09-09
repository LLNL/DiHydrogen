////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/gpu/runtime.hpp"
#include "wait.hpp"

namespace
{

__global__ void wait_kernel(const long long int cycles)
{
  const long long int start = clock64();
  long long int cur;
  do
  {
    cur = clock64();
  } while (cur - start < cycles);
}

} // anonymous namespace

void gpu_wait(double length, h2::gpu::DeviceStream stream)
{
  // Determine and cache the GPU frequency to convert length (in
  // seconds) to cycles. May not be totally accurate, but good enough
  // for these uses.
  static long long int freq_hz = 0;
  if (freq_hz == 0)
  {
    int device = h2::gpu::current_gpu();
    int freq_khz;
#if H2_HAS_CUDA
    H2_CHECK_CUDA(
      cudaDeviceGetAttribute(&freq_khz, cudaDevAttrClockRate, device));
#elif H2_HAS_ROCM
    H2_CHECK_HIP(
      hipDeviceGetAttribute(&freq_khz, hipDeviceAttributeClockRate, device));
#else
#error "Unknown GPU arch"
#endif
    freq_hz = static_cast<long long int>(freq_khz) * 1000ll; // KHz -> Hz
  }
  const long long int cycles = length * freq_hz;

  h2::gpu::launch_kernel(wait_kernel, 1, 1, 0, stream, cycles);
}

void gpu_wait(double length, const h2::ComputeStream& stream)
{
  gpu_wait(length, stream.get_stream<h2::Device::GPU>());
}
