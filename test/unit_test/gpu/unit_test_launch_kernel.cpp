////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>


#include "h2/gpu/runtime.hpp"

#include "../tensor/utils.hpp"

using namespace h2;

extern void
unit_test_gpu_launch_kernel_test(int*, int, const h2::gpu::DeviceStream&);

TEST_CASE("Kernels successfully launch", "[gpu]")
{
  auto stream = ComputeStream{Device::GPU};
  DeviceBuf<int, Device::GPU> buf(1);
  write_ele<Device::GPU>(buf.buf, 0, 0, stream);
  unit_test_gpu_launch_kernel_test(
      buf.buf, 42, stream.get_stream<Device::GPU>());
  REQUIRE(read_ele<Device::GPU>(buf.buf, stream) == 42);
}
