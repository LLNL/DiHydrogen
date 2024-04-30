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

  auto src_stream = ComputeStream{SrcDev};
  auto dst_stream = ComputeStream{DstDev};

  DeviceBuf<DataType, SrcDev> src(buf_size);
  DeviceBuf<DataType, DstDev> dst(buf_size);

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<SrcDev>(src.buf, i, src_val);
    write_ele<DstDev>(dst.buf, i, dst_val);
  }

  REQUIRE_NOTHROW(CopyBuffer(
      dst.buf, dst_stream, src.buf, src_stream, buf_size));

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    // Source is unchanged:
    REQUIRE(read_ele<SrcDev>(src.buf, i) == src_val);
    // Destination has the source value:
    REQUIRE(read_ele<DstDev>(dst.buf, i) == src_val);
  }
}

TEMPLATE_LIST_TEST_CASE("Same-type tensor copy works",
                        "[tensor][copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = Tensor<DataType>;
  using DstTensorType = Tensor<DataType>;
  constexpr DataType src_val = static_cast<DataType>(1);
  constexpr DataType dst_val = static_cast<DataType>(2);

  SECTION("Copying into existing tensor works without resizing")
  {
    SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});
    DstTensorType dst_tensor(DstDev, {4, 6}, {DT::Any, DT::Any});

    DataType* dst_orig_data = dst_tensor.data();

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      write_ele<SrcDev>(src_tensor.data(), i, src_val);
      write_ele<DstDev>(dst_tensor.data(), i, dst_val);
    }

    REQUIRE_NOTHROW(Copy(dst_tensor, src_tensor));

    REQUIRE(dst_tensor.shape() == ShapeTuple{4, 6});
    REQUIRE(dst_tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(dst_tensor.strides() == StrideTuple{1, 4});
    REQUIRE(dst_tensor.numel() == 4 * 6);
    REQUIRE_FALSE(dst_tensor.is_empty());
    REQUIRE(dst_tensor.is_contiguous());
    REQUIRE_FALSE(dst_tensor.is_view());
    REQUIRE(src_tensor.data() != dst_tensor.data());
    REQUIRE(dst_tensor.data() == dst_orig_data);

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      REQUIRE(read_ele<SrcDev>(src_tensor.data(), i) == src_val);
      REQUIRE(read_ele<DstDev>(dst_tensor.data(), i) == src_val);
    }
  }

  SECTION("Copying into different-sized tensor works")
  {
    SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});
    DstTensorType dst_tensor(DstDev, {2, 2}, {DT::Any, DT::Any});

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      write_ele<SrcDev>(src_tensor.data(), i, src_val);
    }
    for (std::size_t i = 0; i < dst_tensor.numel(); ++i)
    {
      write_ele<DstDev>(dst_tensor.data(), i, dst_val);
    }

    REQUIRE_NOTHROW(Copy(dst_tensor, src_tensor));

    REQUIRE(dst_tensor.shape() == ShapeTuple{4, 6});
    REQUIRE(dst_tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(dst_tensor.strides() == StrideTuple{1, 4});
    REQUIRE(dst_tensor.numel() == 4 * 6);
    REQUIRE_FALSE(dst_tensor.is_empty());
    REQUIRE(dst_tensor.is_contiguous());
    REQUIRE_FALSE(dst_tensor.is_view());
    REQUIRE(src_tensor.data() != dst_tensor.data());

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      REQUIRE(read_ele<SrcDev>(src_tensor.data(), i) == src_val);
    }
    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      REQUIRE(read_ele<DstDev>(dst_tensor.data(), i) == src_val);
    }
  }

  SECTION("Copying an empty tensor works")
  {
    SrcTensorType src_tensor(SrcDev);
    DstTensorType dst_tensor(DstDev, {2, 4}, {DT::Any, DT::Any});

    REQUIRE_NOTHROW(Copy(dst_tensor, src_tensor));

    REQUIRE(dst_tensor.is_empty());
  }

  SECTION("Copying non-contiguous tensors works")
  {
    SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});
    DstTensorType dst_tensor(DstDev, {4, 6}, {DT::Any, DT::Any});

    // Resize to be non-contiguous.
    src_tensor.resize(
        src_tensor.shape(), src_tensor.dim_types(), StrideTuple{2, 4});

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      write_ele<DstDev>(dst_tensor.data(), i, dst_val);
    }
    for_ndim(src_tensor.shape(), [&](const ScalarIndexTuple& i) {
      write_ele<SrcDev>(src_tensor.get(i), 0, src_val);
    });

    REQUIRE_NOTHROW(Copy(dst_tensor, src_tensor));

    REQUIRE(dst_tensor.shape() == ShapeTuple{4, 6});
    REQUIRE(dst_tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(dst_tensor.strides() == StrideTuple{2, 4});
    REQUIRE(dst_tensor.numel() == 4 * 6);
    REQUIRE_FALSE(dst_tensor.is_empty());
    REQUIRE_FALSE(dst_tensor.is_contiguous());
    REQUIRE_FALSE(dst_tensor.is_view());
    REQUIRE(src_tensor.data() != dst_tensor.data());

    for_ndim(src_tensor.shape(), [&](const ScalarIndexTuple& i) {
      REQUIRE(read_ele<SrcDev>(src_tensor.get(i)) == src_val);
      REQUIRE(read_ele<DstDev>(dst_tensor.get(i)) == src_val);
    });
  }
}
