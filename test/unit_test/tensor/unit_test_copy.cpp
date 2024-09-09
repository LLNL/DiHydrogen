////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"
#include "h2/tensor/tensor.hpp"
#include "h2/utils/unique_ptr_cast.hpp"
#include "utils.hpp"

#include "../wait.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

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

  SECTION("Copy buffer works with real type")
  {
    REQUIRE_NOTHROW(
      copy_buffer(dst.buf, dst_stream, src.buf, src_stream, buf_size));

    for (std::size_t i = 0; i < buf_size; ++i)
    {
      // Source is unchanged:
      REQUIRE(read_ele<SrcDev>(src.buf, i, src_stream) == src_val);
      // Destination has the source value:
      REQUIRE(read_ele<DstDev>(dst.buf, i, dst_stream) == src_val);
    }
  }

  SECTION("Copy buffer works with void*")
  {
    REQUIRE_NOTHROW(copy_buffer(static_cast<void*>(dst.buf),
                                dst_stream,
                                static_cast<const void*>(src.buf),
                                src_stream,
                                buf_size * sizeof(DataType)));

    for (std::size_t i = 0; i < buf_size; ++i)
    {
      // Source is unchanged:
      REQUIRE(read_ele<SrcDev>(src.buf, i, src_stream) == src_val);
      // Destination has the source value:
      REQUIRE(read_ele<DstDev>(dst.buf, i, dst_stream) == src_val);
    }
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

    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
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

    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
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

    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
    {
      write_ele<SrcDev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
    }
    for (DataIndexType i = 0; i < dst_tensor.numel(); ++i)
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

    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
    {
      REQUIRE(read_ele<SrcDev>(src_tensor.data(), i, src_tensor.get_stream())
              == src_val);
    }
    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
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

    for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
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

TEMPLATE_LIST_TEST_CASE("Same-type tensor copy works with BaseTensor",
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
    SrcTensorType src_tensor_real(SrcDev, {4, 6}, {DT::Sample, DT::Any});
    DstTensorType dst_tensor_real(DstDev, {4, 6}, {DT::Any, DT::Any});
    BaseTensor* src_tensor = &src_tensor_real;
    BaseTensor* dst_tensor = &dst_tensor_real;

    DataType* dst_orig_data = dst_tensor_real.data();

    for (DataIndexType i = 0; i < src_tensor_real.numel(); ++i)
    {
      write_ele<SrcDev>(
        src_tensor_real.data(), i, src_val, src_tensor_real.get_stream());
      write_ele<DstDev>(
        dst_tensor_real.data(), i, dst_val, dst_tensor_real.get_stream());
    }

    REQUIRE_NOTHROW(copy(*dst_tensor, *src_tensor));

    REQUIRE(dst_tensor_real.shape() == ShapeTuple{4, 6});
    REQUIRE(dst_tensor_real.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(dst_tensor_real.strides() == StrideTuple{1, 4});
    REQUIRE(dst_tensor_real.numel() == 4 * 6);
    REQUIRE_FALSE(dst_tensor_real.is_empty());
    REQUIRE(dst_tensor_real.is_contiguous());
    REQUIRE_FALSE(dst_tensor_real.is_view());
    REQUIRE(src_tensor_real.data() != dst_tensor_real.data());
    REQUIRE(dst_tensor_real.data() == dst_orig_data);

    for (DataIndexType i = 0; i < src_tensor_real.numel(); ++i)
    {
      REQUIRE(read_ele<SrcDev>(
                src_tensor_real.data(), i, src_tensor_real.get_stream())
              == src_val);
      REQUIRE(read_ele<DstDev>(
                dst_tensor_real.data(), i, dst_tensor_real.get_stream())
              == src_val);
    }
  }
}

TEMPLATE_LIST_TEST_CASE("make_accessible_on_device works",
                        "[tensor][copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = Tensor<DataType>;
  using DstTensorType = Tensor<DataType>;

  SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});

  std::unique_ptr<DstTensorType> dst_tensor =
    make_accessible_on_device(src_tensor, DstDev);

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

TEMPLATE_LIST_TEST_CASE("make_accessible_on_device works with constant tensors",
                        "[tensor][copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = Tensor<DataType>;
  using DstTensorType = Tensor<DataType>;

  const SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});

  std::unique_ptr<DstTensorType> dst_tensor =
    make_accessible_on_device(src_tensor, DstDev);

  REQUIRE(dst_tensor->shape() == src_tensor.shape());
  REQUIRE(dst_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(dst_tensor->strides() == src_tensor.strides());
  REQUIRE(dst_tensor->get_device() == DstDev);

  if (SrcDev == DstDev)
  {
    // dst_tensor should be a view of src_tensor.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Const);
    REQUIRE(dst_tensor->const_data() == src_tensor.const_data());
  }
#ifdef H2_TEST_WITH_GPU
  else if (gpu::is_integrated())
  {
    // Should also have a view.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Const);
    REQUIRE(dst_tensor->const_data() == src_tensor.const_data());
  }
#endif
  else
  {
    REQUIRE_FALSE(dst_tensor->is_view());
    REQUIRE(dst_tensor->const_data() != src_tensor.const_data());
  }
}

TEMPLATE_LIST_TEST_CASE("make_accessible_on_device works with subviews",
                        "[tensor][copy]",
                        AllDevPairsList)
{
  constexpr Device SrcDev = meta::tlist::At<TestType, 0>::value;
  constexpr Device DstDev = meta::tlist::At<TestType, 1>::value;
  using SrcTensorType = Tensor<DataType>;
  using DstTensorType = Tensor<DataType>;

  SrcTensorType src_tensor(SrcDev, {4, 6}, {DT::Sample, DT::Any});
  std::unique_ptr<SrcTensorType> src_view = src_tensor.view({ALL, IRng{1, 3}});

  std::unique_ptr<DstTensorType> dst_tensor =
    make_accessible_on_device(*src_view, DstDev);

  REQUIRE(dst_tensor->shape() == src_view->shape());
  REQUIRE(dst_tensor->dim_types() == src_view->dim_types());
  REQUIRE(dst_tensor->strides() == src_view->strides());
  REQUIRE(dst_tensor->get_device() == DstDev);

  if (SrcDev == DstDev)
  {
    // dst_tensor should be a view of src_tensor.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Mutable);
    REQUIRE(dst_tensor->data() == src_view->data());
  }
#ifdef H2_TEST_WITH_GPU
  else if (gpu::is_integrated())
  {
    // Should also have a view.
    REQUIRE(dst_tensor->get_view_type() == ViewType::Mutable);
    REQUIRE(dst_tensor->data() == src_view->data());
  }
#endif
  else
  {
    REQUIRE_FALSE(dst_tensor->is_view());
    REQUIRE(dst_tensor->data() != src_view->data());
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
    REQUIRE_NOTHROW(copy_buffer(
      buf_cpu.buf, ComputeStream{Device::CPU}, buf_gpu.buf, stream, buf_size));
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

TEMPLATE_LIST_TEST_CASE("Same-type cast works",
                        "[tensor][copy]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  using TensorType = Tensor<Type>;

  TensorType tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};
  auto cast_tensor = cast<Type>(tensor);

  REQUIRE(cast_tensor->shape() == tensor.shape());
  REQUIRE(cast_tensor->dim_types() == tensor.dim_types());
  REQUIRE(cast_tensor->strides() == tensor.strides());
  REQUIRE(cast_tensor->get_device() == tensor.get_device());
  REQUIRE(cast_tensor->is_view());
  REQUIRE(cast_tensor->get_view_type() == ViewType::Mutable);
  REQUIRE(cast_tensor->data() == tensor.data());
  REQUIRE(cast_tensor->get_type_info() == tensor.get_type_info());
  REQUIRE(cast_tensor->get_type_info() == get_h2_type<Type>());
}

TEMPLATE_LIST_TEST_CASE("Different-type cast works",
                        "[tensor][copy]",
                        AllDevComputeTypePairsPairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using SrcType = meta::tlist::At<meta::tlist::At<TestType, 1>, 0>;
  using DstType = meta::tlist::At<meta::tlist::At<TestType, 1>, 1>;
  using SrcTensorType = Tensor<SrcType>;
  using DstTensorType = Tensor<DstType>;
  constexpr SrcType src_val = static_cast<SrcType>(42);
  constexpr DstType dst_val = static_cast<DstType>(42);

  SrcTensorType src_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    write_ele<Dev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
  }

  std::unique_ptr<DstTensorType> cast_tensor = cast<DstType>(src_tensor);

  REQUIRE(cast_tensor->shape() == src_tensor.shape());
  REQUIRE(cast_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(cast_tensor->strides() == src_tensor.strides());
  REQUIRE(cast_tensor->get_device() == src_tensor.get_device());
  // Since it is a cross-product, SrcType may equal DstType.
  if constexpr (std::is_same_v<SrcType, DstType>)
  {
    REQUIRE(cast_tensor->is_view());
    REQUIRE(cast_tensor->data() == src_tensor.data());
  }
  else
  {
    REQUIRE_FALSE(cast_tensor->is_view());
    REQUIRE(reinterpret_cast<void*>(cast_tensor->data())
            != reinterpret_cast<void*>(src_tensor.data()));
  }
  REQUIRE(cast_tensor->get_type_info() == get_h2_type<DstType>());

  for (DataIndexType i = 0; i < cast_tensor->numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(src_tensor.data(), i, src_tensor.get_stream())
            == src_val);
    REQUIRE(read_ele<Dev>(cast_tensor->data(), i, cast_tensor->get_stream())
            == dst_val);
  }
}

TEMPLATE_LIST_TEST_CASE("Different-type cast works with constant tensors",
                        "[tensor][copy]",
                        AllDevComputeTypePairsPairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using SrcType = meta::tlist::At<meta::tlist::At<TestType, 1>, 0>;
  using DstType = meta::tlist::At<meta::tlist::At<TestType, 1>, 1>;
  using SrcTensorType = Tensor<SrcType>;
  using DstTensorType = Tensor<DstType>;
  constexpr SrcType src_val = static_cast<SrcType>(42);
  constexpr DstType dst_val = static_cast<DstType>(42);

  SrcTensorType src_tensor_orig{Dev, {4, 6}, {DT::Sample, DT::Any}};
  const SrcTensorType& src_tensor = src_tensor_orig;

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    write_ele<Dev>(src_tensor_orig.data(), i, src_val, src_tensor.get_stream());
  }

  std::unique_ptr<DstTensorType> cast_tensor = cast<DstType>(src_tensor);

  REQUIRE(cast_tensor->shape() == src_tensor.shape());
  REQUIRE(cast_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(cast_tensor->strides() == src_tensor.strides());
  REQUIRE(cast_tensor->get_device() == src_tensor.get_device());
  // Since it is a cross-product, SrcType may equal DstType.
  if constexpr (std::is_same_v<SrcType, DstType>)
  {
    REQUIRE(cast_tensor->is_view());
    REQUIRE(cast_tensor->const_data() == src_tensor.const_data());
  }
  else
  {
    REQUIRE_FALSE(cast_tensor->is_view());
    REQUIRE(reinterpret_cast<const void*>(cast_tensor->const_data())
            != reinterpret_cast<const void*>(src_tensor.const_data()));
  }
  REQUIRE(cast_tensor->get_type_info() == get_h2_type<DstType>());

  for (DataIndexType i = 0; i < cast_tensor->numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(src_tensor.const_data(), i, src_tensor.get_stream())
            == src_val);
    REQUIRE(
      read_ele<Dev>(cast_tensor->const_data(), i, cast_tensor->get_stream())
      == dst_val);
  }
}

TEMPLATE_LIST_TEST_CASE("Cast through a BaseTensor works",
                        "[tensor][copy]",
                        AllDevComputeTypePairsPairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using SrcType = meta::tlist::At<meta::tlist::At<TestType, 1>, 0>;
  using DstType = meta::tlist::At<meta::tlist::At<TestType, 1>, 1>;
  using SrcTensorType = Tensor<SrcType>;
  using DstTensorType = Tensor<DstType>;
  constexpr SrcType src_val = static_cast<SrcType>(42);
  constexpr DstType dst_val = static_cast<DstType>(42);

  SrcTensorType src_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    write_ele<Dev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
  }

  BaseTensor& base_tensor = src_tensor;
  std::unique_ptr<DstTensorType> cast_tensor = cast<DstType>(base_tensor);

  REQUIRE(cast_tensor->shape() == src_tensor.shape());
  REQUIRE(cast_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(cast_tensor->strides() == src_tensor.strides());
  REQUIRE(cast_tensor->get_device() == src_tensor.get_device());
  // Since it is a cross-product, SrcType may equal DstType.
  if constexpr (std::is_same_v<SrcType, DstType>)
  {
    REQUIRE(cast_tensor->is_view());
    REQUIRE(cast_tensor->data() == src_tensor.data());
  }
  else
  {
    REQUIRE_FALSE(cast_tensor->is_view());
    REQUIRE(reinterpret_cast<void*>(cast_tensor->data())
            != reinterpret_cast<void*>(src_tensor.data()));
  }
  REQUIRE(cast_tensor->get_type_info() == get_h2_type<DstType>());

  for (DataIndexType i = 0; i < cast_tensor->numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(src_tensor.data(), i, src_tensor.get_stream())
            == src_val);
    REQUIRE(read_ele<Dev>(cast_tensor->data(), i, cast_tensor->get_stream())
            == dst_val);
  }
}

TEMPLATE_LIST_TEST_CASE("Runtime cast through a BaseTensor works",
                        "[tensor][copy]",
                        AllDevComputeTypePairsPairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using SrcType = meta::tlist::At<meta::tlist::At<TestType, 1>, 0>;
  using DstType = meta::tlist::At<meta::tlist::At<TestType, 1>, 1>;
  using SrcTensorType = Tensor<SrcType>;
  using DstTensorType = Tensor<DstType>;
  constexpr SrcType src_val = static_cast<SrcType>(42);
  constexpr DstType dst_val = static_cast<DstType>(42);
  const TypeInfo DstRuntimeType = get_h2_type<DstType>();

  SrcTensorType src_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    write_ele<Dev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
  }

  BaseTensor& base_tensor = src_tensor;
  std::unique_ptr<BaseTensor> cast_base_tensor =
    cast(DstRuntimeType, base_tensor);
  std::unique_ptr<DstTensorType> cast_tensor =
    downcast_uptr<DstTensorType>(cast_base_tensor);

  REQUIRE(cast_tensor->shape() == src_tensor.shape());
  REQUIRE(cast_tensor->dim_types() == src_tensor.dim_types());
  REQUIRE(cast_tensor->strides() == src_tensor.strides());
  REQUIRE(cast_tensor->get_device() == src_tensor.get_device());
  // Since it is a cross-product, SrcType may equal DstType.
  if constexpr (std::is_same_v<SrcType, DstType>)
  {
    REQUIRE(cast_tensor->is_view());
    REQUIRE(cast_tensor->data() == src_tensor.data());
  }
  else
  {
    REQUIRE_FALSE(cast_tensor->is_view());
    REQUIRE(reinterpret_cast<void*>(cast_tensor->data())
            != reinterpret_cast<void*>(src_tensor.data()));
  }
  REQUIRE(cast_tensor->get_type_info() == get_h2_type<DstType>());

  for (DataIndexType i = 0; i < cast_tensor->numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(src_tensor.data(), i, src_tensor.get_stream())
            == src_val);
    REQUIRE(read_ele<Dev>(cast_tensor->data(), i, cast_tensor->get_stream())
            == dst_val);
  }
}
