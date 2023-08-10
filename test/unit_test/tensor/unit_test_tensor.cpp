 ////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <type_traits>

#include "h2/tensor/tensor_cpu.hpp"

using namespace h2;

// Catch2's TEMPLATE_TEST_CASE does not support non-type template
// parameters. We therefore turn them into types this way to simplify
// things. These declarations are just to save typing.
using CPUDev_t = std::integral_constant<Device, Device::CPU>;

// Placeholder for now, the data type of the tensor.
using DataType = float;

TEMPLATE_TEST_CASE("Tensors can be created", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataType, TestType::value>;
  REQUIRE_NOTHROW(TensorType());
  REQUIRE_NOTHROW(TensorType({2}, {DT::Any}));

  DataType* null_buf = nullptr;
  REQUIRE_NOTHROW(TensorType(null_buf, {0}, {DT::Any}, {1}));
  REQUIRE_NOTHROW(TensorType(const_cast<const DataType*>(null_buf), {0}, {DT::Any}, {1}));
}

TEMPLATE_TEST_CASE("Tensor metadata is sane", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  REQUIRE(tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(tensor.shape(0) == 4);
  REQUIRE(tensor.shape(1) == 6);
  REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor.dim_type(0) == DT::Sample);
  REQUIRE(tensor.dim_type(1) == DT::Any);
  REQUIRE(tensor.strides() == StrideTuple{1, 4});
  REQUIRE(tensor.ndim() == 2);
  REQUIRE(tensor.numel() == 4*6);
  REQUIRE_FALSE(tensor.is_empty());
  REQUIRE(tensor.is_contiguous());
  REQUIRE_FALSE(tensor.is_view());
  REQUIRE(tensor.get_device() == TestType::value);
  REQUIRE(tensor.data() != nullptr);
  REQUIRE(tensor.const_data() != nullptr);
}

TEMPLATE_TEST_CASE("Empty tensor metadata is sane", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType();

  REQUIRE(tensor.shape() == ShapeTuple{});
  REQUIRE(tensor.dim_types() == DTTuple{});
  REQUIRE(tensor.strides() == StrideTuple{});
  REQUIRE(tensor.ndim() == 0);
  REQUIRE(tensor.numel() == 0);
  REQUIRE(tensor.is_empty());
  REQUIRE(tensor.is_contiguous());
  REQUIRE_FALSE(tensor.is_view());
  REQUIRE(tensor.get_device() == TestType::value);
  REQUIRE(tensor.data() == nullptr);
  REQUIRE(tensor.const_data() == nullptr);
}

TEMPLATE_TEST_CASE("Resizing tensors works", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  SECTION("Resizing without changing the number of dimensions") {
    tensor.resize({2, 3});
    REQUIRE(tensor.shape() == ShapeTuple{2, 3});
    REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(tensor.strides() == StrideTuple{1, 2});
    REQUIRE(tensor.ndim() == 2);
    REQUIRE(tensor.numel() == 2*3);
    REQUIRE(tensor.is_contiguous());
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(tensor.const_data() != nullptr);
  }
  SECTION("Resizing while adding dimensions") {
    tensor.resize({2, 3, 4}, {DT::Sample, DT::Sequence, DT::Any});
    REQUIRE(tensor.shape() == ShapeTuple{2, 3, 4});
    REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Sequence, DT::Any});
    REQUIRE(tensor.strides() == StrideTuple{1, 2, 6});
    REQUIRE(tensor.ndim() == 3);
    REQUIRE(tensor.numel() == 2*3*4);
    REQUIRE(tensor.is_contiguous());
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(tensor.const_data() != nullptr);
  }
  SECTION("Resizing while adding dimensions requires dim types") {
    REQUIRE_THROWS(tensor.resize({2, 3, 4}));
  }
  SECTION("Emptying tensors") {
    tensor.empty();
    REQUIRE(tensor.shape() == ShapeTuple{});
    REQUIRE(tensor.dim_types() == DTTuple{});
    REQUIRE(tensor.strides() == StrideTuple{});
    REQUIRE(tensor.ndim() == 0);
    REQUIRE(tensor.numel() == 0);
    REQUIRE(tensor.is_empty());
    REQUIRE(tensor.is_contiguous());
    REQUIRE(tensor.data() == nullptr);
    REQUIRE(tensor.const_data() == nullptr);

    SECTION("Empty tensors can be resized") {
      tensor.resize({2, 3}, {DT::Sample, DT::Any});
      REQUIRE(tensor.shape() == ShapeTuple{2, 3});
      REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
      REQUIRE(tensor.strides() == StrideTuple{1, 2});
      REQUIRE(tensor.ndim() == 2);
      REQUIRE(tensor.numel() == 2*3);
      REQUIRE(tensor.is_contiguous());
      REQUIRE(tensor.data() != nullptr);
      REQUIRE(tensor.const_data() != nullptr);
    }
  }
}

TEMPLATE_TEST_CASE("Writing to tensors works", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataIndexType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  DataIndexType* buf = tensor.data();
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    buf[i] = i;
  }

  DataIndexType idx = 0;
  for (DimType j = 0; j < 6; ++j)
  {
    for (DimType i = 0; i < 4; ++i)
    {
      REQUIRE(tensor.get({i, j}) == idx);
      ++idx;
    }
  }
}

TEMPLATE_TEST_CASE("Attaching tensors to existing buffers works", "[tensor]", CPUDev_t) {
  using TensorType = Tensor<DataType, TestType::value>;

  // Even if DataType is floating point, small integers are exact.
  DataType buf[4*6];
  for (std::size_t i = 0; i < 4*6; ++i)
  {
    buf[i] = i;
  }

  TensorType tensor = TensorType(buf, {4, 6}, {DT::Sample, DT::Any}, {1, 4});
  REQUIRE(tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor.strides() == StrideTuple{1, 4});
  REQUIRE(tensor.ndim() == 2);
  REQUIRE(tensor.numel() == 4*6);
  REQUIRE(tensor.is_view());

  for (std::size_t i = 0; i < 4 * 6; ++i)
  {
    REQUIRE(tensor.data()[i] == i);
  }
  DataIndexType idx = 0;
  for (DimType j = 0; j < 6; ++j)
  {
    for (DimType i = 0; i < 4; ++i)
    {
      REQUIRE(tensor.get({i, j}) == idx);
      ++idx;
    }
  }
}

TEMPLATE_TEST_CASE("Viewing tensors works", "[tensor]", CPUDev_t)
{
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < 4*6; ++i)
  {
    tensor.data()[i] = i;
  }

  SECTION("Basic views work")
  {
    TensorType* view = tensor.view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 4 * 6);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());

    for (DataIndexType i = 0; i < 4*6; ++i)
    {
      REQUIRE(view->data()[i] == i);
    }
  }
  SECTION("Constant views work")
  {
    TensorType* view = tensor.const_view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 4 * 6);
    REQUIRE(view->is_view());
    REQUIRE(view->const_data() == tensor.data());
    REQUIRE(view->is_contiguous());
  }
  SECTION("Viewing a subtensor works")
  {
    TensorType* view = tensor.view({DRng(1), ALL});
    REQUIRE(view->shape() == ShapeTuple{6});
    REQUIRE(view->dim_types() == DTTuple{DT::Any});
    REQUIRE(view->strides() == StrideTuple{4});
    REQUIRE(view->ndim() == 1);
    REQUIRE(view->numel() == 6);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == (tensor.data() + 1));
    REQUIRE_FALSE(view->is_contiguous());

    for (DimType i = 0; i < 6; ++i)
    {
      REQUIRE(view->get({i}) == (1 + 4*i));
    }
  }
  SECTION("Operator-style views work")
  {
    TensorType* view = tensor({DRng(1, 3), ALL});
    REQUIRE(view->shape() == ShapeTuple{2, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 2 * 6);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == (tensor.data() + 1));
    REQUIRE_FALSE(view->is_contiguous());

    for (DimType j = 0; j < 6; ++j)
    {
      for (DimType i = 0; i < 2; ++i)
      {
        REQUIRE(view->get({i, j}) == (i+1 + j*4));
      }
    }
  }
  SECTION("Viewing a view works")
  {
    TensorType* view_orig = tensor.view();
    TensorType* view = view_orig->view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 4 * 6);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());

    for (DataIndexType i = 0; i < 4*6; ++i)
    {
      REQUIRE(view->data()[i] == i);
    }
  }
  SECTION("Unviewing a view works")
  {
    TensorType* view = tensor.view();
    REQUIRE(view->is_view());
    view->unview();
    REQUIRE_FALSE(view->is_view());
    REQUIRE(view->shape() == ShapeTuple{});
    REQUIRE(view->dim_types() == DTTuple{});
    REQUIRE(view->strides() == StrideTuple{});
    REQUIRE(view->ndim() == 0);
    REQUIRE(view->numel() == 0);
    REQUIRE(view->is_empty());
    REQUIRE(view->data() == nullptr);
  }
  SECTION("Emptying a view unviews")
  {
    TensorType* view = tensor.view();
    view->empty();
    REQUIRE_FALSE(view->is_view());
    REQUIRE(view->shape() == ShapeTuple{});
    REQUIRE(view->dim_types() == DTTuple{});
    REQUIRE(view->strides() == StrideTuple{});
    REQUIRE(view->ndim() == 0);
    REQUIRE(view->numel() == 0);
    REQUIRE(view->is_empty());
    REQUIRE(view->data() == nullptr);
  }
#ifdef H2_DEBUG
  SECTION("Unviewing a non-view fails")
  {
    REQUIRE_THROWS(tensor.unview());
  }
#endif
  SECTION("Resizing a view fails")
  {
    TensorType* view = tensor.view();
    REQUIRE_THROWS(view->resize(ShapeTuple{2, 3}));
  }
  SECTION("Viewing an invalid range fails")
  {
    REQUIRE_THROWS(tensor.view({ALL, DRng(0, 7)}));
  }
}

// contiguous is not yet implemented.
TEMPLATE_TEST_CASE("Making tensors contiguous works", "[tensor][!mayfail]", CPUDev_t)
{
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < 4 * 6; ++i)
  {
    tensor.data()[i] = i;
  }
  TensorType* view = tensor.view({DRng(1), ALL});
  REQUIRE_FALSE(view->is_contiguous());
  TensorType* contig = view->contiguous();
  REQUIRE(contig->is_contiguous());
  REQUIRE(contig->data() != view->data());
  REQUIRE(contig->shape() == ShapeTuple{6});
  REQUIRE(contig->strides() == StrideTuple{1});
  for (DimType i = 0; i < 6; ++i)
  {
    REQUIRE(contig->get(i) == (1 + 4 * i));
  }
}
