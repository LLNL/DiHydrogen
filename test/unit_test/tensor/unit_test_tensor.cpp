 ////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "h2/tensor/tensor.hpp"
#include "utils.hpp"

using namespace h2;

TEST_CASE("last(FixedSizeTuple)", "[utilities]")
{
    using TupleType = h2::FixedSizeTuple<int, unsigned, 8U>;
    SECTION("last of a zero-element (empty) tuple is error")
    {
        TupleType x;
        CHECK_THROWS(h2::last(x));
    }

    SECTION("last of a single-element tuple returns only element")
    {
        TupleType x = {1};
        CHECK(h2::last(x) == 1);
    }

    SECTION("last of multi-element tuple returns last element")
    {
        TupleType x = {1, 2, 3, 4, 5};
        CHECK(h2::last(x) == 5);
    }
}

TEST_CASE("init(FixedSizeTuple)", "[utilities]")
{
    using TupleType = h2::FixedSizeTuple<int, unsigned, 8U>;
    SECTION("init of a zero-element (empty) tuple is error")
    {
        TupleType x;
        CHECK_THROWS(h2::init(x));
    }

    SECTION("init of a single-element tuple is empty")
    {
        TupleType x = {1};
        CHECK(h2::init(x).size() == 0);
    }

    SECTION("init of multi-element tuple returns first n-1 elements")
    {
        TupleType x = {1, 2, 3, 4, 5};
        CHECK(h2::init(x) == TupleType{1, 2, 3, 4});
    }
}

TEST_CASE("is_shape_contained", "[tensor]")
{
  REQUIRE(is_shape_contained(CoordTuple{DRng{0, 2}, ALL}, ShapeTuple{4, 4}));
  REQUIRE_FALSE(
      is_shape_contained(CoordTuple{DRng{2, 6}, ALL}, ShapeTuple{4, 4}));
}

TEMPLATE_LIST_TEST_CASE("Tensors can be created", "[tensor]", AllDevList)
{
  using TensorType = Tensor<DataType, TestType::value>;
  REQUIRE_NOTHROW(TensorType());
  REQUIRE_NOTHROW(TensorType({2}, {DT::Any}));

  DataType* null_buf = nullptr;
  REQUIRE_NOTHROW(TensorType(null_buf, {0}, {DT::Any}, {1}));
  REQUIRE_NOTHROW(TensorType(const_cast<const DataType*>(null_buf), {0}, {DT::Any}, {1}));
}

TEMPLATE_LIST_TEST_CASE("Tensor metadata is sane", "[tensor]", AllDevList)
{
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  REQUIRE(tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(tensor.shape(0) == 4);
  REQUIRE(tensor.shape(1) == 6);
  REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor.dim_type(0) == DT::Sample);
  REQUIRE(tensor.dim_type(1) == DT::Any);
  REQUIRE(tensor.strides() == StrideTuple{1, 4});
  REQUIRE(tensor.stride(0) == 1);
  REQUIRE(tensor.stride(1) == 4);
  REQUIRE(tensor.ndim() == 2);
  REQUIRE(tensor.numel() == 4*6);
  REQUIRE_FALSE(tensor.is_empty());
  REQUIRE(tensor.is_contiguous());
  REQUIRE_FALSE(tensor.is_view());
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::None);
  REQUIRE(tensor.get_device() == TestType::value);
  REQUIRE(tensor.data() != nullptr);
  REQUIRE(tensor.const_data() != nullptr);
  REQUIRE(tensor.get({0, 0}) == tensor.data());

  const TensorType const_tensor = TensorType({4, 6}, {DT::Sample, DT::Any});
  REQUIRE(const_tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(const_tensor.shape(0) == 4);
  REQUIRE(const_tensor.shape(1) == 6);
  REQUIRE(const_tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(const_tensor.dim_type(0) == DT::Sample);
  REQUIRE(const_tensor.dim_type(1) == DT::Any);
  REQUIRE(const_tensor.strides() == StrideTuple{1, 4});
  REQUIRE(const_tensor.stride(0) == 1);
  REQUIRE(const_tensor.stride(1) == 4);
  REQUIRE(const_tensor.ndim() == 2);
  REQUIRE(const_tensor.numel() == 4*6);
  REQUIRE_FALSE(const_tensor.is_empty());
  REQUIRE(const_tensor.is_contiguous());
  REQUIRE_FALSE(const_tensor.is_view());
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::None);
  REQUIRE(const_tensor.get_device() == TestType::value);
  REQUIRE(const_tensor.data() != nullptr);
  REQUIRE(const_tensor.const_data() != nullptr);
  REQUIRE(const_tensor.get({0, 0}) == const_tensor.data());
}

TEMPLATE_LIST_TEST_CASE("Empty tensor metadata is sane", "[tensor]", AllDevList)
{
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
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::None);
  REQUIRE(tensor.get_device() == TestType::value);
  REQUIRE(tensor.data() == nullptr);
  REQUIRE(tensor.const_data() == nullptr);
}

TEMPLATE_LIST_TEST_CASE("Resizing tensors works", "[tensor]", AllDevList)
{
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

TEMPLATE_LIST_TEST_CASE("Writing to tensors works", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  DataType* buf = tensor.data();
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(buf, i, static_cast<DataType>(i));
  }

  DataIndexType idx = 0;
  for (DimType j = 0; j < tensor.shape(1); ++j)
  {
    for (DimType i = 0; i < tensor.shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(tensor.get({i, j})) == idx);
      ++idx;
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Attaching tensors to existing buffers works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType, TestType::value>;
  constexpr std::size_t buf_size = 4*6;

  DeviceBuf<DataType, Dev> buf(buf_size);
  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<Dev>(buf.buf, i, static_cast<DataType>(i));
  }

  TensorType tensor = TensorType(buf.buf, {4, 6}, {DT::Sample, DT::Any}, {1, 4});
  REQUIRE(tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor.strides() == StrideTuple{1, 4});
  REQUIRE(tensor.ndim() == 2);
  REQUIRE(tensor.numel() == buf_size);
  REQUIRE(tensor.is_view());
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::Mutable);

  for (std::size_t i = 0; i < tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(tensor.data(), i) == i);
  }
  DataIndexType idx = 0;
  for (DimType j = 0; j < tensor.shape(1); ++j)
  {
    for (DimType i = 0; i < tensor.shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(tensor.get({i, j})) == idx);
      ++idx;
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Viewing tensors works", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(tensor.data(), i, static_cast<DataType>(i));
  }

  SECTION("Basic views work")
  {
    TensorType* view = tensor.view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == tensor.numel());
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());

    for (DataIndexType i = 0; i < view->numel(); ++i)
    {
      REQUIRE(read_ele<Dev>(view->data(), i) == i);
    }
  }
  SECTION("Constant views work")
  {
    TensorType* view = tensor.const_view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == tensor.numel());
    REQUIRE(view->is_view());
    REQUIRE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Const);
    REQUIRE(view->const_data() == tensor.data());
    REQUIRE_THROWS(view->data());
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

    for (DimType i = 0; i < view->shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(view->get({i})) == (1 + 4*i));
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

    for (DimType j = 0; j < view->shape(1); ++j)
    {
      for (DimType i = 0; i < view->shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(view->get({i, j})) == (i+1 + j*4));
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
    REQUIRE(view->numel() == tensor.numel());
    REQUIRE(view->is_view());
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());

    for (DataIndexType i = 0; i < view->numel(); ++i)
    {
      REQUIRE(read_ele<Dev>(view->data(), i) == i);
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
    REQUIRE_THROWS(view->resize(ShapeTuple{2, 3, 4},
                                DTTuple{DT::Sample, DT::Any, DT::Any}));
  }
  SECTION("Viewing an invalid range fails")
  {
    REQUIRE_THROWS(tensor.view({ALL, DRng(0, 7)}));
  }
}

TEMPLATE_LIST_TEST_CASE("Viewing constant tensors works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType, TestType::value>;

  const TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});

  SECTION("Basic views work")
  {
    TensorType* view = tensor.view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == tensor.numel());
    REQUIRE(view->is_view());
    REQUIRE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Const);
    REQUIRE(view->const_data() == tensor.data());
    REQUIRE_THROWS(view->data());
    REQUIRE(view->is_contiguous());
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
    REQUIRE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Const);
    REQUIRE(view->const_data() == (tensor.data() + 1));
    REQUIRE_THROWS(view->data());
    REQUIRE_FALSE(view->is_contiguous());
  }
}

// contiguous is not yet implemented.
TEMPLATE_LIST_TEST_CASE("Making tensors contiguous works",
                        "[tensor][!mayfail]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType, TestType::value>;

  TensorType tensor = TensorType({4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(tensor.data(), i, static_cast<DataType>(i));
  }

  SECTION("Making contiguous tensors contiguous works")
  {
    REQUIRE(tensor.is_contiguous());
    TensorType* contig = tensor.contiguous();
    REQUIRE(contig->is_view());
    REQUIRE(contig->data() == tensor.data());
  }
  SECTION("Making non-contiguous tensors contiguous work")
  {
    TensorType* view = tensor.view({DRng(1), ALL});
    REQUIRE_FALSE(view->is_contiguous());

    TensorType* contig = view->contiguous();
    REQUIRE(contig->is_contiguous());
    REQUIRE(contig->data() != view->data());
    REQUIRE(contig->shape() == ShapeTuple{6});
    REQUIRE(contig->strides() == StrideTuple{1});
    for (DimType i = 0; i < contig->shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(contig->get({i})) == (1 + 4*i));
    }
  }
}
