////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/tensor/tensor.hpp"
#include "h2/tensor/copy.hpp"
#include "utils.hpp"

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Buffer copy works", "[tensor][copy]", AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  constexpr std::size_t buf_size = 32;
  constexpr DataType src_val = static_cast<DataType>(1);
  constexpr DataType dst_val = static_cast<DataType>(2);

  auto src_sync = SyncInfo<SrcDev>{};
  auto dst_sync = SyncInfo<DstDev>{};

  DeviceBuf<DataType, SrcDev> src(buf_size);
  DeviceBuf<DataType, DstDev> dst(buf_size);

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<SrcDev>(src.buf, i, src_val);
    write_ele<DstDev>(dst.buf, i, dst_val);
  }

  REQUIRE_NOTHROW(CopyBuffer<DstDev, SrcDev>(
      dst.buf, dst_sync, src.buf, src_sync, buf_size));

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    // Source is unchanged:
    REQUIRE(read_ele<SrcDev>(src.buf, i) == src_val);
    // Destination has the source value:
    REQUIRE(read_ele<DstDev>(dst.buf, i) == src_val);
  }
}
