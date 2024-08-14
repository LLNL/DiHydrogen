////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/gpu/runtime.hpp"


namespace
{

__global__ void test_kernel(int* buf, int val)
{
  *buf = val;
}

}

void unit_test_gpu_launch_kernel_test(int* buf,
                                      int val,
                                      const h2::gpu::DeviceStream& stream)
{
  h2::gpu::launch_kernel(test_kernel, 1, 1, 0, stream, buf, val);
}
