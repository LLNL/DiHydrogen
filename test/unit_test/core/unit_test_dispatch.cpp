////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/core/dispatch.hpp"

#include "../tensor/utils.hpp"

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Static dispatch works for compute types",
                        "[dispatch]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using DispatchType = meta::tlist::At<TestType, 1>;

  DeviceBuf<DispatchType, Dev> buf{1};
  buf.fill(static_cast<DispatchType>(0));
  if constexpr (Dev == Device::CPU)
  {
    dispatch_test(Dev, buf.buf);
    REQUIRE(buf.buf[0] == 42);
  }
#ifdef H2_TEST_WITH_GPU
  else if constexpr (Dev == Device::GPU)
  {
    // Does nothing on GPUs.
    dispatch_test(Dev, buf.buf);
    REQUIRE(read_ele<Device::GPU>(buf.buf, 0) == 42);
  }
#endif
}

template <>
void h2::impl::dispatch_test_impl<bool>(CPUDev_t, bool* v)
{
  *v = true;
}

// This is HAS_GPU rather than TEST_WITH_GPU because dispatch_test will
// always generate dispatch code for the GPU version whenever H2 has
// GPU support.
#ifdef H2_HAS_GPU
template <>
void h2::impl::dispatch_test_impl<bool>(GPUDev_t, bool* v)
{
  write_ele<Device::GPU, bool>(v, 0, true, 0);
}
#endif

TEMPLATE_LIST_TEST_CASE("Static dispatch works for new types",
                        "[dispatch]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DispatchType = bool;

  DeviceBuf<DispatchType, Dev> buf{1};
  buf.fill(static_cast<DispatchType>(false));
  dispatch_test(Dev, buf.buf);
  REQUIRE(read_ele<Dev>(buf.buf, 0) == true);
}
