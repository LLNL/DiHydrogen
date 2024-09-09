////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/tensor/init/fill.hpp"
#include "utils.hpp"


using namespace h2;


TEMPLATE_LIST_TEST_CASE("Zeroing buffers works",
                        "[tensor][fill]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  constexpr std::size_t buf_size = 32;

  auto stream = ComputeStream{Dev};
  DeviceBuf<Type, Dev> buf(buf_size);

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    write_ele<Dev>(buf.buf, i, static_cast<Type>(42), stream);
  }

  REQUIRE_NOTHROW(zero(buf.buf, stream, buf_size));

  for (std::size_t i = 0; i < buf_size; ++i)
  {
    REQUIRE(read_ele<Dev>(buf.buf, i, stream) == static_cast<Type>(0));
  }
}

TEMPLATE_LIST_TEST_CASE("Zeroing contiguous tensors",
                        "[tensor][fill]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  using TensorType = Tensor<Type>;

  TensorType tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
        tensor.data(), i, static_cast<Type>(42), tensor.get_stream());
  }

  REQUIRE_NOTHROW(zero(tensor));

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(tensor.data(), i, tensor.get_stream())
            == static_cast<Type>(0));
  }
}

TEMPLATE_LIST_TEST_CASE("Zeroing contiguous tensors through BaseTensor",
                        "[tensor][fill]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  using TensorType = Tensor<Type>;

  TensorType tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};
  BaseTensor& base_tensor = tensor;

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
        tensor.data(), i, static_cast<Type>(42), tensor.get_stream());
  }

  REQUIRE_NOTHROW(zero(base_tensor));

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(tensor.data(), i, tensor.get_stream())
            == static_cast<Type>(0));
  }
}

TEMPLATE_LIST_TEST_CASE("Filling contiguous tensors",
                        "[tensor][fill]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  using TensorType = Tensor<Type>;
  constexpr Type fill_val = static_cast<Type>(42);

  TensorType tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    write_ele<Dev>(
        tensor.data(), i, static_cast<Type>(0), tensor.get_stream());
  }

  REQUIRE_NOTHROW(fill(tensor, fill_val));

  for (DataIndexType i = 0; i < tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(tensor.data(), i, tensor.get_stream()) == fill_val);
  }
}
