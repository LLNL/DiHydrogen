////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"

using namespace h2;

namespace scalar2range_tuple_tests
{
static_assert(scalar2range_tuple(ScalarIndexTuple{}) == IndexRangeTuple{},
              "scalar2range_tuple is wrong");
static_assert(scalar2range_tuple(ScalarIndexTuple{0})
                  == IndexRangeTuple{DRng{0}},
              "scalar2range_tuple is wrong");
static_assert(scalar2range_tuple(ScalarIndexTuple{0, 3})
                  == IndexRangeTuple{DRng{0}, DRng{3}},
              "scalar2range_tuple is wrong");
}  // namespace scalar2range_tuple_tests

namespace get_index_range_start_tests
{
constexpr IndexRangeTuple empty_range;
static_assert(get_index_range_start(empty_range).size() == 0,
              "get_index_range_start of empty range is not empty");

constexpr IndexRangeTuple coords1(1, 2);
constexpr auto coords1_range_start = get_index_range_start(coords1);
static_assert(coords1_range_start.size() == 2,
              "get_index_range_start returned wrong size");
static_assert(coords1_range_start[0] == 1,
              "get_index_range_start returned wrong start");
static_assert(coords1_range_start[1] == 2,
              "get_index_range_start returned wrong start");

constexpr IndexRangeTuple coords2(DRng{1, 4}, ALL);
constexpr auto coords2_range_start = get_index_range_start(coords2);
static_assert(coords2_range_start.size() == 2,
              "get_index_range_start returned wrong size");
static_assert(coords2_range_start[0] == 1,
              "get_index_range_start returned wrong start");
static_assert(coords2_range_start[1] == 0,
              "get_index_range_start returned wrong start");
}  // namespace get_index_range_start_tests

namespace is_index_range_empty_tests
{
static_assert(is_index_range_empty(IndexRangeTuple{}),
              "is_index_range_empty is wrong");
static_assert(is_index_range_empty(IndexRangeTuple{DRng{}}),
              "is_index_range_empty is wrong");
static_assert(is_index_range_empty(IndexRangeTuple{DRng{0, 2}, DRng{}}),
              "is_index_range_empty is wrong");
static_assert(!is_index_range_empty(IndexRangeTuple{DRng{0}}),
              "is_index_range_empty is wrong");
static_assert(!is_index_range_empty(IndexRangeTuple{DRng{0, 2}, ALL}),
              "is_index_range_empty is wrong");
static_assert(!is_index_range_empty(IndexRangeTuple{ALL}),
              "is_index_range_empty is wrong");
}

namespace get_index_range_shape_tests
{
constexpr ShapeTuple empty_shape;
constexpr IndexRangeTuple empty_coord;
constexpr auto empty_range_shape = get_index_range_shape(empty_coord, empty_shape);
static_assert(empty_range_shape.size() == 0,
              "get_index_range_shape not empty");

constexpr ShapeTuple shape1(8, 8);
constexpr IndexRangeTuple coord1(DRng(0, 2), DRng(0, 3));
constexpr auto range_shape1 = get_index_range_shape(coord1, shape1);
static_assert(range_shape1.size() == 2,
              "get_index_range_shape returned wrong size");
static_assert(range_shape1[0] == 2,
              "get_index_range_shape index 0 is wrong");
static_assert(range_shape1[1] == 3,
              "get_index_range_shape index 1 is wrong");

constexpr IndexRangeTuple coord2(DRng(0, 2), ALL);
constexpr auto range_shape2 = get_index_range_shape(coord2, shape1);
static_assert(range_shape2.size() == 2,
              "get_index_range_shape returned wrong size");
static_assert(range_shape2[0] == 2,
              "get_index_range_shape index 0 is wrong");
static_assert(range_shape2[1] == 8,
              "get_index_range_shape index 1 is wrong");

constexpr IndexRangeTuple coord3(ALL);
constexpr auto range_shape3 = get_index_range_shape(coord3, shape1);
static_assert(range_shape3.size() == 2,
              "get_index_range_shape returned wrong size");
static_assert(range_shape3[0] == 8,
              "get_index_range_shape index 0 is wrong");
static_assert(range_shape3[1] == 8,
              "get_index_range_shape index 1 is wrong");

constexpr IndexRangeTuple coord4(DRng(0));
constexpr auto range_shape4 = get_index_range_shape(coord4, shape1);
static_assert(range_shape4.size() == 1,
              "get_index_range_shape returned wrong size");
static_assert(range_shape4[0] == 8,
              "get_index_range_shape index 0 is wrong");

constexpr IndexRangeTuple coord5(DRng(0), DRng(2));
constexpr auto range_shape5 = get_index_range_shape(coord5, shape1);
static_assert(range_shape5.size() == 0,
              "get_index_range_shape returned wrong size");

constexpr IndexRangeTuple coord6(DRng(0), DRng(2, 4));
constexpr auto range_shape6 = get_index_range_shape(coord6, shape1);
static_assert(range_shape6.size() == 1,
              "get_index_range_shape returned wrong size");
static_assert(range_shape6[0] == 2, "get_index_range_shape index 0 is wrong");
}  // get_index_range_shape_tests

namespace is_shape_contained_tests
{
constexpr ShapeTuple shape(3, 2, 1);

constexpr IndexRangeTuple coord1(ALL, ALL, ALL);
static_assert(is_index_range_contained(coord1, shape),
              "is_index_range_contained is wrong");

constexpr IndexRangeTuple coord2(DRng(0, 3), DRng(0, 2), DRng(0, 1));
static_assert(is_index_range_contained(coord2, shape),
              "is_index_range_contained is wrong");

constexpr IndexRangeTuple coord3(DRng(0, 4), ALL, ALL);
static_assert(!is_index_range_contained(coord3, shape),
              "is_index_range_contained is wrong");

constexpr IndexRangeTuple coord4(ALL, ALL, DRng(1, 2));
static_assert(!is_index_range_contained(coord4, shape),
              "is_index_range_contained is wrong");

constexpr IndexRangeTuple coord5(DRng(1, 3), ALL);
static_assert(is_index_range_contained(coord5, shape),
              "is_index_range_contained is wrong");
}
