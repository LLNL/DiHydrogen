////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/tensor.hpp"
#include "h2/utils/unique_ptr_cast.hpp"

#include "../tensor/utils.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

TEMPLATE_LIST_TEST_CASE("unique_ptr_cast works with tensors",
                        "[utilities][unique_ptr_cast]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using Type = meta::tlist::At<TestType, 1>;
  using TensorType = Tensor<Type>;

  std::unique_ptr<BaseTensor> base_ptr = std::make_unique<TensorType>(
    Dev, ShapeTuple{4, 6}, DTTuple{DT::Sample, DT::Any});
  void* orig_data = base_ptr->storage_data();
  std::unique_ptr<TensorType> derived_ptr;
  REQUIRE_NOTHROW([&]() {
    derived_ptr = downcast_uptr<TensorType>(base_ptr);
  }());
  // Some additional sanity-checks:
  REQUIRE(derived_ptr->shape() == ShapeTuple{4, 6});
  REQUIRE(orig_data == static_cast<void*>(derived_ptr->data()));
}
