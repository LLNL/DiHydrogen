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
#include "../wait.hpp"

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
    write_ele<SrcDev>(src.buf, i, src_val, src_stream);
    write_ele<DstDev>(dst.buf, i, dst_val, dst_stream);
  }

  REQUIRE_NOTHROW(copy_buffer(
      dst.buf, dst_stream, src.buf, src_stream, buf_size));

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    // Source is unchanged:
    REQUIRE(read_ele<SrcDev>(src.buf, i, src_stream) == src_val);
    // Destination has the source value:
    REQUIRE(read_ele<DstDev>(dst.buf, i, dst_stream) == src_val);
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
      write_ele<SrcDev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
      write_ele<DstDev>(dst_tensor.data(), i, dst_val, dst_tensor.get_stream());
    }

    REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

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
      REQUIRE(read_ele<SrcDev>(src_tensor.data(), i, src_tensor.get_stream())
              == src_val);
      REQUIRE(read_ele<DstDev>(dst_tensor.data(), i, dst_tensor.get_stream())
              == src_val);
    }
  }

  SECTION("Copying into different-sized tensor works")
  {
    SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});
    DstTensorType dst_tensor(DstDev, {2, 2}, {DT::Any, DT::Any});

    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      write_ele<SrcDev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
    }
    for (std::size_t i = 0; i < dst_tensor.numel(); ++i)
    {
      write_ele<DstDev>(dst_tensor.data(), i, dst_val, dst_tensor.get_stream());
    }

    REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

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
      REQUIRE(read_ele<SrcDev>(src_tensor.data(), i, src_tensor.get_stream())
              == src_val);
    }
    for (std::size_t i = 0; i < src_tensor.numel(); ++i)
    {
      REQUIRE(read_ele<DstDev>(dst_tensor.data(), i, dst_tensor.get_stream())
              == src_val);
    }
  }

  SECTION("Copying an empty tensor works")
  {
    SrcTensorType src_tensor(SrcDev);
    DstTensorType dst_tensor(DstDev, {2, 4}, {DT::Any, DT::Any});

    REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

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
      write_ele<DstDev>(dst_tensor.data(), i, dst_val, dst_tensor.get_stream());
    }
    for_ndim(src_tensor.shape(), [&](const ScalarIndexTuple& i) {
      write_ele<SrcDev>(src_tensor.get(i), 0, src_val, src_tensor.get_stream());
    });

    REQUIRE_NOTHROW(copy(dst_tensor, src_tensor));

    REQUIRE(dst_tensor.shape() == ShapeTuple{4, 6});
    REQUIRE(dst_tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(dst_tensor.strides() == StrideTuple{2, 4});
    REQUIRE(dst_tensor.numel() == 4 * 6);
    REQUIRE_FALSE(dst_tensor.is_empty());
    REQUIRE_FALSE(dst_tensor.is_contiguous());
    REQUIRE_FALSE(dst_tensor.is_view());
    REQUIRE(src_tensor.data() != dst_tensor.data());

    for_ndim(src_tensor.shape(), [&](const ScalarIndexTuple& i) {
      REQUIRE(read_ele<SrcDev>(src_tensor.get(i), src_tensor.get_stream())
              == src_val);
      REQUIRE(read_ele<DstDev>(dst_tensor.get(i), dst_tensor.get_stream())
              == src_val);
    });
  }
}

TEMPLATE_LIST_TEST_CASE("MakeAccessibleOnDevice works",
                        "[tensor][copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = Tensor<DataType>;
  using DstTensorType = Tensor<DataType>;

  SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});

  auto dst_tensor = make_accessible_on_device(src_tensor, DstDev);

  REQUIRE(dst_tensor->shape() == src_tensor.shape());
  REQUIRE(dst_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(dst_tensor->strides() == src_tensor.strides());
  REQUIRE(dst_tensor->get_device() == DstDev);

  if (SrcDev == DstDev)
  {
    // dst_tensor should be a view of src_tensor.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Mutable);
    REQUIRE(dst_tensor->data() == src_tensor.data());
  }
#ifdef H2_TEST_WITH_GPU
  else if (gpu::is_integrated())
  {
    // Should also have a view.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Mutable);
    REQUIRE(dst_tensor->data() == src_tensor.data());
  }
#endif
  else
  {
    REQUIRE_FALSE(dst_tensor->is_view());
    REQUIRE(dst_tensor->data() != src_tensor.data());
  }
}

#ifdef H2_TEST_WITH_GPU

TEST_CASE("GPU-GPU copy synchronizes correctly", "[tensor][copy]")
{
  // We ping-pong a buffer between two streams and if they do not sync
  // correctly, we may get an incorrect buffer.
  constexpr std::size_t buf_size = 16;
  constexpr std::size_t change_i = 1;

  ComputeStream stream1 = create_new_compute_stream<Device::GPU>();
  ComputeStream stream2 = create_new_compute_stream<Device::GPU>();

  DeviceBuf<DataType, Device::GPU> buf1{buf_size};
  DeviceBuf<DataType, Device::GPU> buf2{buf_size};

  // Run this a few times since sync issues don't always show up.
  for (int iter = 0; iter < 10; ++iter)
  {
    buf1.fill(static_cast<DataType>(1));
    buf2.fill(static_cast<DataType>(2));
    gpu_wait(0.001, stream1);
    write_ele_nosync<Device::GPU>(
        buf1.buf, change_i, static_cast<DataType>(3), stream1);
    REQUIRE_NOTHROW(
        copy_buffer(buf2.buf, stream2, buf1.buf, stream1, buf_size));
    // read_ele syncs appropriately.
    for (std::size_t i = 0; i < buf_size; ++i)
    {
      auto v1 = read_ele<Device::GPU>(buf1.buf, i, stream1);
      auto v2 = read_ele<Device::GPU>(buf2.buf, i, stream2);
      if (i == change_i)
      {
        REQUIRE(v1 == static_cast<DataType>(3));
        REQUIRE(v2 == static_cast<DataType>(3));
      }
      else
      {
        REQUIRE(v1 == static_cast<DataType>(1));
        REQUIRE(v2 == static_cast<DataType>(1));
      }
    }
  }
}

TEST_CASE("GPU-CPU copy synchronzies correctly", "[tensor][copy]")
{
  // We attempt to copy a buffer from the GPU to the CPU.
  // If the CPU copy doesn't sync with the stream correctly, it may get
  // an old buffer.
  constexpr std::size_t buf_size = 16;
  constexpr std::size_t change_i = 1;

  ComputeStream stream = create_new_compute_stream<Device::GPU>();

  DeviceBuf<DataType, Device::GPU> buf_gpu{buf_size};
  DeviceBuf<DataType, Device::CPU> buf_cpu{buf_size};

  // Run a few times.
  for (int iter = 0; iter < 10; ++iter)
  {
    buf_gpu.fill(static_cast<DataType>(1));
    buf_cpu.fill(static_cast<DataType>(2));
    gpu_wait(0.001, stream);
    write_ele_nosync<Device::GPU>(
        buf_gpu.buf, change_i, static_cast<DataType>(3), stream);
    REQUIRE_NOTHROW(copy_buffer(buf_cpu.buf,
                                ComputeStream{Device::CPU},
                                buf_gpu.buf,
                                stream,
                                buf_size));
    stream.wait_for_this();
    // Verify only the CPU buffer.
    for (std::size_t i = 0; i < buf_size; ++i)
    {
      if (i == change_i)
      {
        REQUIRE(buf_cpu.buf[i] == static_cast<DataType>(3));
      }
      else
      {
        REQUIRE(buf_cpu.buf[i] == static_cast<DataType>(1));
      }
    }
  }
}

#endif  // H2_TEST_WITH_GPU
