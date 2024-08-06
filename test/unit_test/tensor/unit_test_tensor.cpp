////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <memory>

#include "h2/tensor/tensor.hpp"
#include "h2/utils/typename.hpp"
#include "utils.hpp"

using namespace h2;

TEST_CASE("FixedSizeTuple::back", "[utilities]")
{
  using TupleType = h2::FixedSizeTuple<int, unsigned, 8U>;
#ifdef H2_DEBUG
  SECTION("back of a zero-element (empty) tuple is error")
  {
    TupleType x;
    CHECK_THROWS(x.back());
  }
#endif

  SECTION("back of a single-element tuple returns only element")
  {
    TupleType x = {1};
    CHECK(x.back() == 1);
  }

  SECTION("back of multi-element tuple returns last element")
  {
    TupleType x = {1, 2, 3, 4, 5};
    CHECK(x.back() == 5);
  }
}

TEST_CASE("init(FixedSizeTuple)", "[utilities]")
{
    using TupleType = h2::FixedSizeTuple<int, unsigned, 8U>;
#ifdef H2_DEBUG
    SECTION("init of a zero-element (empty) tuple is error")
    {
        TupleType x;
        CHECK_THROWS(h2::init(x));
    }
#endif

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

TEST_CASE("is_index_range_contained", "[tensor]")
{
  REQUIRE(is_index_range_contained(IndexRangeTuple{IRng{0, 2}, ALL},
                                   ShapeTuple{4, 4}));
  REQUIRE_FALSE(is_index_range_contained(IndexRangeTuple{IRng{2, 6}, ALL},
                                         ShapeTuple{4, 4}));
}

TEST_CASE("for_ndim", "[tensor]")
{
  SECTION("Empty")
  {
    ShapeTuple shape;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(false);  // Should never be reached.
    });
  }

  SECTION("1d")
  {
    ShapeTuple shape{4};
    std::vector<ScalarIndexTuple> v = {{0}, {1}, {2}, {3}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    });
  }

  SECTION("1 with start")
  {
    ShapeTuple shape{4};
    std::vector<ScalarIndexTuple> v = {{1}, {2}, {3}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    }, {1});
  }

  SECTION("2d")
  {
    ShapeTuple shape{4, 2};
    std::vector<ScalarIndexTuple> v = {
        {0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    });
  }

  SECTION("2d with start")
  {
    ShapeTuple shape{4, 2};
    std::vector<ScalarIndexTuple> v = {{2, 1}, {3, 1}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    }, {2, 1});
  }

  SECTION("3d")
  {
    ShapeTuple shape{2, 2, 2};
    std::vector<ScalarIndexTuple> v = {{0, 0, 0},
                                       {1, 0, 0},
                                       {0, 1, 0},
                                       {1, 1, 0},
                                       {0, 0, 1},
                                       {1, 0, 1},
                                       {0, 1, 1},
                                       {1, 1, 1}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    });
  }

  SECTION("3d with start")
  {
    ShapeTuple shape{2, 2, 2};
    std::vector<ScalarIndexTuple> v = {{1, 0, 1}, {0, 1, 1}, {1, 1, 1}};
    DataIndexType i = 0;
    for_ndim(shape, [&](ScalarIndexTuple c) {
      REQUIRE(c == v[i]);
      ++i;
    }, {1, 0, 1});
  }

  SECTION("Out-of-range start works")
  {
    ShapeTuple shape{4, 2};
    for_ndim(shape,
             [&](ScalarIndexTuple c) { FAIL("Should not be executed"); },
             {4, 2});
  }
}

TEMPLATE_LIST_TEST_CASE("Tensors can be created", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;
  REQUIRE_NOTHROW(TensorType(Dev));
  REQUIRE_NOTHROW(TensorType(Dev, {2}, {DT::Any}));

  DataType* null_buf = nullptr;
  REQUIRE_NOTHROW(
      TensorType(Dev, null_buf, {0}, {DT::Any}, {1}, ComputeStream{Dev}));
  REQUIRE_NOTHROW(TensorType(Dev,
                             const_cast<const DataType*>(null_buf),
                             {0},
                             {DT::Any},
                             {1},
                             ComputeStream{Dev}));
}

TEMPLATE_LIST_TEST_CASE("Tensor metadata is sane", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});

  REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
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
  REQUIRE(tensor.const_get({0, 0}) == tensor.data());
  REQUIRE_FALSE(tensor.is_lazy());

  const TensorType const_tensor =
      TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});
  REQUIRE(const_tensor.get_type_info() == get_h2_type<DataType>());
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
  REQUIRE_FALSE(const_tensor.is_const_view());
  REQUIRE(const_tensor.get_view_type() == ViewType::None);
  REQUIRE(const_tensor.get_device() == TestType::value);
  REQUIRE(const_tensor.data() != nullptr);
  REQUIRE(const_tensor.const_data() != nullptr);
  REQUIRE(const_tensor.get({0, 0}) == const_tensor.data());
  REQUIRE(const_tensor.const_get({0, 0}) == const_tensor.data());
  REQUIRE_FALSE(const_tensor.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("Base tensor metadata is sane",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  std::unique_ptr<BaseTensor> tensor = std::make_unique<TensorType>(
      Dev, ShapeTuple{4, 6}, DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor->get_type_info() == get_h2_type<DataType>());
  REQUIRE(tensor->shape() == ShapeTuple{4, 6});
  REQUIRE(tensor->shape(0) == 4);
  REQUIRE(tensor->shape(1) == 6);
  REQUIRE(tensor->dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor->dim_type(0) == DT::Sample);
  REQUIRE(tensor->dim_type(1) == DT::Any);
  REQUIRE(tensor->strides() == StrideTuple{1, 4});
  REQUIRE(tensor->stride(0) == 1);
  REQUIRE(tensor->stride(1) == 4);
  REQUIRE(tensor->ndim() == 2);
  REQUIRE(tensor->numel() == 4*6);
  REQUIRE_FALSE(tensor->is_empty());
  REQUIRE(tensor->is_contiguous());
  REQUIRE_FALSE(tensor->is_view());
  REQUIRE_FALSE(tensor->is_const_view());
  REQUIRE(tensor->get_view_type() == ViewType::None);
  REQUIRE(tensor->get_device() == TestType::value);

  std::unique_ptr<const BaseTensor> const_tensor =
      std::make_unique<const TensorType>(
          Dev, ShapeTuple{4, 6}, DTTuple{DT::Sample, DT::Any});
  REQUIRE(const_tensor->get_type_info() == get_h2_type<DataType>());
  REQUIRE(const_tensor->shape() == ShapeTuple{4, 6});
  REQUIRE(const_tensor->shape(0) == 4);
  REQUIRE(const_tensor->shape(1) == 6);
  REQUIRE(const_tensor->dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(const_tensor->dim_type(0) == DT::Sample);
  REQUIRE(const_tensor->dim_type(1) == DT::Any);
  REQUIRE(const_tensor->strides() == StrideTuple{1, 4});
  REQUIRE(const_tensor->stride(0) == 1);
  REQUIRE(const_tensor->stride(1) == 4);
  REQUIRE(const_tensor->ndim() == 2);
  REQUIRE(const_tensor->numel() == 4*6);
  REQUIRE_FALSE(const_tensor->is_empty());
  REQUIRE(const_tensor->is_contiguous());
  REQUIRE_FALSE(const_tensor->is_view());
  REQUIRE_FALSE(const_tensor->is_const_view());
  REQUIRE(const_tensor->get_view_type() == ViewType::None);
  REQUIRE(const_tensor->get_device() == TestType::value);
}

TEMPLATE_LIST_TEST_CASE("Empty tensor metadata is sane", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev);

  REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
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
  REQUIRE_FALSE(tensor.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("Single element tensor metadata is sane",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {1}, {DT::Any});

  REQUIRE(tensor.get_type_info() == get_h2_type<DataType>());
  REQUIRE(tensor.shape() == ShapeTuple{1});
  REQUIRE(tensor.dim_types() == DTTuple{DT::Any});
  REQUIRE(tensor.strides() == StrideTuple{1});
  REQUIRE(tensor.ndim() == 1);
  REQUIRE(tensor.numel() == 1);
  REQUIRE_FALSE(tensor.is_empty());
  REQUIRE(tensor.is_contiguous());
  REQUIRE_FALSE(tensor.is_view());
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::None);
  REQUIRE(tensor.get_device() == TestType::value);
  REQUIRE(tensor.data() != nullptr);
  REQUIRE(tensor.const_data() != nullptr);
  REQUIRE(tensor.get({0}) == tensor.data());
  REQUIRE(tensor.const_get({0}) == tensor.data());
  REQUIRE_FALSE(tensor.is_lazy());
}

TEMPLATE_LIST_TEST_CASE("Lazy and strict tensors are sane",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  REQUIRE_FALSE(TensorType(Dev).is_lazy());
  REQUIRE_FALSE(TensorType(Dev, StrictAlloc).is_lazy());
  REQUIRE(TensorType(Dev, LazyAlloc).is_lazy());

  REQUIRE_FALSE(
      TensorType(Dev, {4, 6}, {DT::Sample, DT::Any}, StrictAlloc).is_lazy());
  REQUIRE(TensorType(Dev, {4, 6}, {DT::Sample, DT::Any}, LazyAlloc).is_lazy());
}

TEMPLATE_LIST_TEST_CASE("Resizing tensors works", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});

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
  SECTION("Resizing while changing strides works")
  {
    tensor.resize({4, 6}, {DT::Sample, DT::Any}, {1, 8});
    REQUIRE(tensor.shape() == ShapeTuple{4, 6});
    REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(tensor.strides() == StrideTuple{1, 8});
    REQUIRE(tensor.ndim() == 2);
    REQUIRE(tensor.numel() == 4*6);
    REQUIRE_FALSE(tensor.is_contiguous());
    REQUIRE(tensor.data() != nullptr);
    REQUIRE(tensor.const_data() != nullptr);
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
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});

  DataType* buf = tensor.data();
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(buf, i, static_cast<DataType>(i), tensor.get_stream());
  }

  DataIndexType idx = 0;
  for (DimType j = 0; j < tensor.shape(1); ++j)
  {
    for (DimType i = 0; i < tensor.shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(tensor.get({i, j}), tensor.get_stream()) == idx);
      ++idx;
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Attaching tensors to existing buffers works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;
  constexpr std::size_t buf_size = 4*6;

  DeviceBuf<DataType, Dev> buf(buf_size);
  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<Dev>(buf.buf, i, static_cast<DataType>(i), ComputeStream{Dev});
  }

  TensorType tensor = TensorType(
      Dev, buf.buf, {4, 6}, {DT::Sample, DT::Any}, {1, 4}, ComputeStream{Dev});
  REQUIRE(tensor.shape() == ShapeTuple{4, 6});
  REQUIRE(tensor.dim_types() == DTTuple{DT::Sample, DT::Any});
  REQUIRE(tensor.strides() == StrideTuple{1, 4});
  REQUIRE(tensor.ndim() == 2);
  REQUIRE(tensor.numel() == buf_size);
  REQUIRE(tensor.is_view());
  REQUIRE_FALSE(tensor.is_const_view());
  REQUIRE(tensor.get_view_type() == ViewType::Mutable);

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(tensor.data(), i, tensor.get_stream()) == i);
  }
  DataIndexType idx = 0;
  for (DimType j = 0; j < tensor.shape(1); ++j)
  {
    for (DimType i = 0; i < tensor.shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(tensor.get({i, j}), tensor.get_stream()) == idx);
      ++idx;
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Viewing tensors works", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
      tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
  }

  SECTION("Basic views work")
  {
    std::unique_ptr<TensorType> view = tensor.view();
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
      REQUIRE(read_ele<Dev>(view->data(), i, view->get_stream()) == i);
    }
  }
  SECTION("Constant views work")
  {
    std::unique_ptr<TensorType> view = tensor.const_view();
    REQUIRE(view->shape() == ShapeTuple{4, 6});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == tensor.numel());
    REQUIRE(view->is_view());
    REQUIRE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Const);
    REQUIRE(view->const_data() == tensor.data());
    REQUIRE(view->const_get({0, 0}) == tensor.data());
    REQUIRE_THROWS(view->data());
    REQUIRE_THROWS(view->get({0, 0}));
    REQUIRE(view->is_contiguous());
  }
  SECTION("Viewing a (1, ALL) subtensor works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(1), ALL});
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
      REQUIRE(read_ele<Dev>(view->get({i}), view->get_stream()) == (1 + 4*i));
    }
  }
  SECTION("Viewing a (ALL, (1, 3)) subtensor works")
  {
    std::unique_ptr<TensorType> view = tensor({ALL, IRng(1, 3)});
    REQUIRE(view->shape() == ShapeTuple{4, 2});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 4});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 4 * 2);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == (tensor.data() + 4));
    REQUIRE(view->is_contiguous());
    for (DimType j = 0; j < view->shape(1); ++j)
    {
      for (DimType i = 0; i < view->shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(view->get({i, j}), view->get_stream())
                == (i + (j+1) * 4));
      }
    }
  }
  SECTION("Operator-style views work")
  {
    std::unique_ptr<TensorType> view = tensor({IRng(1, 3), ALL});
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
        REQUIRE(read_ele<Dev>(view->get({i, j}), view->get_stream())
                == (i+1 + j*4));
      }
    }
  }
  SECTION("View of a single element works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(1, 2), IRng(0, 1)});
    REQUIRE(view->shape() == ShapeTuple{1, 1});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample, DT::Any});
    REQUIRE(view->strides() == StrideTuple{1, 1});
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == 1);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == (tensor.data() + 1));
    REQUIRE(view->is_contiguous());

    REQUIRE(read_ele<Dev>(view->get({0, 0}), view->get_stream()) == 1);
  }
  SECTION("View of a single element, eliminating dimensions, works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(1), IRng(0)});
    REQUIRE(view->shape() == ShapeTuple{1});
    REQUIRE(view->dim_types() == DTTuple{DT::Scalar});
    REQUIRE(view->strides() == StrideTuple{1});
    REQUIRE(view->ndim() == 1);
    REQUIRE(view->numel() == 1);
    REQUIRE(view->is_view());
    REQUIRE(view->data() == (tensor.data() + 1));
    REQUIRE(view->is_contiguous());

    REQUIRE(read_ele<Dev>(view->get({0}), view->get_stream()) == 1);
  }
  SECTION("Viewing a view works")
  {
    std::unique_ptr<TensorType> view_orig = tensor.view();
    std::unique_ptr<TensorType> view = view_orig->view();
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
      REQUIRE(read_ele<Dev>(view->data(), i, view->get_stream()) == i);
    }
  }
  SECTION("Viewing a subview works")
  {
    std::unique_ptr<TensorType> view_orig = tensor.view({ALL, IRng{1, 3}});
    std::unique_ptr<TensorType> view = view_orig->view();
    REQUIRE(view->shape() == view_orig->shape());
    REQUIRE(view->dim_types() == view_orig->dim_types());
    REQUIRE(view->strides() == view_orig->strides());
    REQUIRE(view->ndim() == 2);
    REQUIRE(view->numel() == view_orig->numel());
    REQUIRE(view->is_view());
    REQUIRE(view->data() == view_orig->data());
    REQUIRE(view->is_contiguous());

    for (DimType j = 0; j < view->shape(1); ++j)
    {
      for (DimType i = 0; i < view->shape(0); ++i)
      {
        REQUIRE(read_ele<Dev>(view->get({i, j}), view->get_stream())
                == (i + (j + 1) * 4));
      }
    }
  }
  SECTION("Unviewing a view works")
  {
    std::unique_ptr<TensorType> view = tensor.view();
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
    std::unique_ptr<TensorType> view = tensor.view();
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
    std::unique_ptr<TensorType> view = tensor.view();
    REQUIRE_THROWS(view->resize(ShapeTuple{2, 3}));
    REQUIRE_THROWS(view->resize(ShapeTuple{2, 3, 4},
                                DTTuple{DT::Sample, DT::Any, DT::Any}));
  }
  SECTION("Viewing an invalid range fails")
  {
    REQUIRE_THROWS(tensor.view({ALL, IRng(0, 7)}));
  }
}

TEMPLATE_LIST_TEST_CASE("Viewing a tensor with a single element works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {1}, {DT::Sample});
  write_ele<Dev>(
    tensor.data(), 0, static_cast<DataType>(1), tensor.get_stream());

  SECTION("Basic views work")
  {
    std::unique_ptr<TensorType> view = tensor.view();
    REQUIRE(view->shape() == ShapeTuple{1});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample});
    REQUIRE(view->strides() == StrideTuple{1});
    REQUIRE(view->ndim() == 1);
    REQUIRE(view->numel() == 1);
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());
    REQUIRE(read_ele<Dev>(view->data(), 0, view->get_stream()) == 1);
  }

  SECTION("Manually-specified view range works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(0, 1)});
    REQUIRE(view->shape() == ShapeTuple{1});
    REQUIRE(view->dim_types() == DTTuple{DT::Sample});
    REQUIRE(view->strides() == StrideTuple{1});
    REQUIRE(view->ndim() == 1);
    REQUIRE(view->numel() == 1);
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());
    REQUIRE(read_ele<Dev>(view->data(), 0, view->get_stream()) == 1);
  }

  SECTION("View with a scalar index works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(0)});
    REQUIRE(view->shape() == ShapeTuple{1});
    REQUIRE(view->dim_types() == DTTuple{DT::Scalar});
    REQUIRE(view->strides() == StrideTuple{1});
    REQUIRE(view->ndim() == 1);
    REQUIRE(view->numel() == 1);
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == tensor.data());
    REQUIRE(view->is_contiguous());
    REQUIRE(read_ele<Dev>(view->data(), 0, view->get_stream()) == 1);
  }
}

TEMPLATE_LIST_TEST_CASE("Viewing constant tensors works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  const TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});

  SECTION("Basic views work")
  {
    std::unique_ptr<TensorType> view = tensor.view();
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
    std::unique_ptr<TensorType> view = tensor({IRng(1, 3), ALL});
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

TEMPLATE_LIST_TEST_CASE("Empty views work", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
      tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
  }

  SECTION("View with fully empty coordinates work")
  {
    std::unique_ptr<TensorType> view = tensor.view(IndexRangeTuple{});
    REQUIRE(view->shape() == ShapeTuple{});
    REQUIRE(view->dim_types() == DTTuple{});
    REQUIRE(view->strides() == StrideTuple{});
    REQUIRE(view->ndim() == 0);
    REQUIRE(view->numel() == 0);
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == nullptr);
    REQUIRE(view->is_contiguous());
  }

  SECTION("View with one coordinate empty works")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(0, 1), IRng()});
    REQUIRE(view->shape() == ShapeTuple{});
    REQUIRE(view->dim_types() == DTTuple{});
    REQUIRE(view->strides() == StrideTuple{});
    REQUIRE(view->ndim() == 0);
    REQUIRE(view->numel() == 0);
    REQUIRE(view->is_view());
    REQUIRE_FALSE(view->is_const_view());
    REQUIRE(view->get_view_type() == ViewType::Mutable);
    REQUIRE(view->data() == nullptr);
    REQUIRE(view->is_contiguous());
  }
}

// contiguous is not yet implemented.
TEMPLATE_LIST_TEST_CASE("Making tensors contiguous works",
                        "[tensor][!mayfail]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
      tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
  }

  SECTION("Making contiguous tensors contiguous works")
  {
    REQUIRE(tensor.is_contiguous());
    std::unique_ptr<TensorType> contig = tensor.contiguous();
    REQUIRE(contig->is_view());
    REQUIRE(contig->data() == tensor.data());
  }
  SECTION("Making non-contiguous tensors contiguous work")
  {
    std::unique_ptr<TensorType> view = tensor.view({IRng(1, 2), ALL});
    REQUIRE_FALSE(view->is_contiguous());

    std::unique_ptr<TensorType> contig = view->contiguous();
    REQUIRE(contig->is_contiguous());
    REQUIRE(contig->data() != view->data());
    REQUIRE(contig->shape() == ShapeTuple{6});
    REQUIRE(contig->strides() == StrideTuple{1});
    for (DimType i = 0; i < contig->shape(0); ++i)
    {
      REQUIRE(read_ele<Dev>(contig->get({i}), contig->get_stream())
              == (1 + 4*i));
    }
  }
}

TEMPLATE_LIST_TEST_CASE("Cloning tensors works",
                        "[tensor]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  TensorType tensor = TensorType(Dev, {4, 6}, {DT::Sample, DT::Any});
  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
      tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
  }

  SECTION("Cloning an entire tensor works")
  {
    std::unique_ptr<TensorType> clone = tensor.clone();
    REQUIRE(clone->shape() == tensor.shape());
    REQUIRE(clone->dim_types() == tensor.dim_types());
    REQUIRE(clone->strides() == tensor.strides());
    REQUIRE(clone->ndim() == tensor.ndim());
    REQUIRE(clone->numel() == tensor.numel());
    REQUIRE(clone->is_empty() == tensor.is_empty());
    REQUIRE_FALSE(clone->is_view());
    REQUIRE_FALSE(clone->is_const_view());
    REQUIRE(clone->get_view_type() == ViewType::None);
    REQUIRE(clone->is_contiguous() == tensor.is_contiguous());
    REQUIRE(clone->get_device() == tensor.get_device());
    REQUIRE(clone->is_lazy() == tensor.is_lazy());
    REQUIRE(clone->data() != nullptr);
    REQUIRE(clone->data() != tensor.data());
  }
}

TEMPLATE_LIST_TEST_CASE("Tensor get/set stream works", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  ComputeStream stream1 = create_new_compute_stream<Dev>();
  ComputeStream stream2 = create_new_compute_stream<Dev>();

  SECTION("Get/set on regular tensor")
  {
    TensorType tensor(Dev, {3, 5}, {DT::Any, DT::Any}, StrictAlloc, stream1);
    REQUIRE(tensor.get_stream() == stream1);
    tensor.set_stream(stream2);
    REQUIRE(tensor.get_stream() == stream2);
  }

  SECTION("Get/set on view")
  {
    TensorType tensor(Dev, {3, 5}, {DT::Any, DT::Any}, StrictAlloc, stream1);
    auto view = tensor.view();
    ComputeStream stream3 = create_new_compute_stream<Dev>();
    // Changing the original should not impact the view.
    REQUIRE(tensor.get_stream() == stream1);
    REQUIRE(view->get_stream() == stream1);
    tensor.set_stream(stream2);
    REQUIRE(tensor.get_stream() == stream2);
    REQUIRE(view->get_stream() == stream1);
    view->set_stream(stream3);
    REQUIRE(view->get_stream() == stream3);
    REQUIRE(tensor.get_stream() == stream2);
  }
}

TEMPLATE_LIST_TEST_CASE("Tensors are printable", "[tensor]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  std::stringstream dev_ss;
  dev_ss << TestType::value;

  std::stringstream ss;

  TensorType tensor(Dev, {3, 5}, {DT::Sample, DT::Any});

  SECTION("No view")
  {
    ss << tensor;
    REQUIRE(ss.str()
            == std::string("Tensor<") + TypeName<DataType>() + ", "
                   + dev_ss.str() + ">(Sample:3 x Any:5)");
  }

  SECTION("View")
  {
    std::unique_ptr<TensorType> view = tensor.view();
    ss << *view;
    REQUIRE(ss.str()
            == std::string("Tensor<") + TypeName<DataType>() + ", "
                   + dev_ss.str() + ">(View of Sample:3 x Any:5)");
  }
}
