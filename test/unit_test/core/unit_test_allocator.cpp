////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/core/allocator.hpp"

#include "../tensor/utils.hpp"
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Allocation and deallocation works",
                        "[allocator]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;

  ComputeStream stream{Dev};

  DataType* buf = h2::internal::Allocator<DataType, Dev>::allocate(8, stream);
  REQUIRE(buf != nullptr);

  // Ensure a second allocation gets a new buffer.
  DataType* buf2 = h2::internal::Allocator<DataType, Dev>::allocate(8, stream);
  REQUIRE(buf2 != nullptr);
  REQUIRE(buf != buf2);

  REQUIRE_NOTHROW(
    h2::internal::Allocator<DataType, Dev>::deallocate(buf, stream));
  REQUIRE_NOTHROW(
    h2::internal::Allocator<DataType, Dev>::deallocate(buf2, stream));
}

TEMPLATE_LIST_TEST_CASE("Zero-size allocation and deallocation works",
                        "[allocator]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;

  ComputeStream stream{Dev};

  // Should succeed, but can't say anything about the pointer.
  DataType* buf = h2::internal::Allocator<DataType, Dev>::allocate(0, stream);
  REQUIRE_NOTHROW(
    h2::internal::Allocator<DataType, Dev>::deallocate(buf, stream));
}
