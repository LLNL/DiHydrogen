////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/strided_memory.hpp"

using namespace h2;

namespace get_contiguous_strides_tests
{
constexpr ShapeTuple shape1;
constexpr auto strides1 = get_contiguous_strides(shape1);
static_assert(strides1.size() == 0);

constexpr ShapeTuple shape2(2, 2);
constexpr auto strides2 = get_contiguous_strides(shape2);
static_assert(strides2.size() == 2);
static_assert(strides2[0] == 1);
static_assert(strides2[1] == 2);

constexpr ShapeTuple shape3(2, 2, 2);
constexpr auto strides3 = get_contiguous_strides(shape3);
static_assert(strides3.size() == 3);
static_assert(strides3[0] == 1);
static_assert(strides3[1] == 2);
static_assert(strides3[2] == 4);

} // namespace get_contiguous_strides_tests

namespace are_strides_contiguous_tests
{
constexpr ShapeTuple shape1;
constexpr StrideTuple strides1;
static_assert(are_strides_contiguous(shape1, strides1));

constexpr ShapeTuple shape2(3);
constexpr StrideTuple strides2(1);
static_assert(are_strides_contiguous(shape2, strides2));

constexpr ShapeTuple shape3(2, 2);
constexpr StrideTuple strides3(1, 2);
static_assert(are_strides_contiguous(shape3, strides3));

constexpr ShapeTuple shape4(2, 2, 2);
constexpr StrideTuple strides4(1, 2, 4);
static_assert(are_strides_contiguous(shape4, strides4));

constexpr StrideTuple strides5(2, 2, 4);
static_assert(!are_strides_contiguous(shape4, strides5));

constexpr StrideTuple strides6(1, 4, 4);
static_assert(!are_strides_contiguous(shape4, strides6));

} // namespace are_strides_contiguous_tests
