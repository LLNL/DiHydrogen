////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/io.hpp"
#include "h2/tensor/tensor.hpp"
#include "utils.hpp"

#include <sstream>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Printing tensors works", "[tensor][io]", AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<DataType>;

  SECTION("Printing an empty tensor works")
  {
    TensorType tensor{Dev};
    std::stringstream ss;
    print(ss, tensor);
    REQUIRE(ss.str() == "[]");
  }

  SECTION("Printing a 1D tensor works")
  {
    TensorType tensor{Dev, {4}, {DT::Any}};
    for (DataIndexType i = 0; i < tensor.numel(); ++i)
    {
      write_ele<Dev>(
        tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
    }
    std::stringstream ss;
    print(ss, tensor);
    REQUIRE(ss.str() == "[0, 1, 2, 3]");
  }

  SECTION("Printing a 2D tensor works")
  {
    TensorType tensor{Dev, {2, 4}, {DT::Any, DT::Any}};
    for (DataIndexType i = 0; i < tensor.numel(); ++i)
    {
      write_ele<Dev>(
        tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
    }
    std::stringstream ss;
    print(ss, tensor);
    char const* expected = "[\n"
                           " [0, 2, 4, 6]\n"
                           " [1, 3, 5, 7]\n"
                           "]";
    REQUIRE(ss.str() == expected);
  }

  SECTION("Printing a 3D tensor works")
  {
    TensorType tensor{Dev, {2, 2, 4}, {DT::Any, DT::Any, DT::Any}};
    for (DataIndexType i = 0; i < tensor.numel(); ++i)
    {
      write_ele<Dev>(
        tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
    }
    std::stringstream ss;
    print(ss, tensor);
    char const* expected = "[\n"
                           " [\n"
                           "  [0, 4, 8, 12]\n"
                           "  [2, 6, 10, 14]\n"
                           " ]\n"
                           " [\n"
                           "  [1, 5, 9, 13]\n"
                           "  [3, 7, 11, 15]\n"
                           " ]\n"
                           "]";
    REQUIRE(ss.str() == expected);
  }

  SECTION("Printing single-element tensors works")
  {
    char const* expected[] = {"[1]",
                              "[\n [1]\n]",
                              "[\n [\n  [1]\n ]\n]",
                              "[\n [\n  [\n   [1]\n  ]\n ]\n]",
                              "[\n [\n  [\n   [\n    [1]\n   ]\n  ] \n]\n]"};
    for (typename ShapeTuple::type ndims = 1; ndims <= 3; ++ndims)
    {
      TensorType tensor{Dev,
                        ShapeTuple{TuplePad<ShapeTuple>(ndims, 1)},
                        DTTuple{TuplePad<DTTuple>(ndims, DT::Any)}};
      write_ele<Dev>(
        tensor.data(), 0, static_cast<DataType>(1), tensor.get_stream());
      std::stringstream ss;
      print(ss, tensor);
      REQUIRE(ss.str() == expected[ndims - 1]);
    }
  }

  SECTION("Printing a contiguous view works")
  {
    TensorType tensor{Dev, {2, 2, 4}, {DT::Any, DT::Any, DT::Any}};
    for (DataIndexType i = 0; i < tensor.numel(); ++i)
    {
      write_ele<Dev>(
        tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
    }
    auto view = tensor.view();
    std::stringstream ss;
    print(ss, *view);
    char const* expected = "[\n"
                           " [\n"
                           "  [0, 4, 8, 12]\n"
                           "  [2, 6, 10, 14]\n"
                           " ]\n"
                           " [\n"
                           "  [1, 5, 9, 13]\n"
                           "  [3, 7, 11, 15]\n"
                           " ]\n"
                           "]";
    REQUIRE(ss.str() == expected);
  }

  SECTION("Printing a sub-view works")
  {
    TensorType tensor{Dev, {2, 4}, {DT::Any, DT::Any}};
    for (DataIndexType i = 0; i < tensor.numel(); ++i)
    {
      write_ele<Dev>(
        tensor.data(), i, static_cast<DataType>(i), tensor.get_stream());
    }
    auto view = tensor.view({ALL, IRng(1, 3)});
    std::stringstream ss;
    print(ss, *view);
    char const* expected = "[\n"
                           " [2, 4]\n"
                           " [3, 5]\n"
                           "]";
    REQUIRE(ss.str() == expected);
  }
}
