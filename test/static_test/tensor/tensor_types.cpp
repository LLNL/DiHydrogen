////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"

using namespace h2;

namespace scalar2range_tuple_tests
{
static_assert(scalar2range_tuple(ScalarIndexTuple{}) == IndexRangeTuple{});
static_assert(scalar2range_tuple(ScalarIndexTuple{0})
              == IndexRangeTuple{IRng{0}});
static_assert(scalar2range_tuple(ScalarIndexTuple{0, 3})
              == IndexRangeTuple{IRng{0}, IRng{3}});
}  // namespace scalar2range_tuple_tests

namespace get_index_range_start_tests
{
constexpr IndexRangeTuple empty_range;
static_assert(get_index_range_start(empty_range).size() == 0);

constexpr IndexRangeTuple coords1(1, 2);
constexpr auto coords1_range_start = get_index_range_start(coords1);
static_assert(coords1_range_start.size() == 2);
static_assert(coords1_range_start[0] == 1);
static_assert(coords1_range_start[1] == 2);

constexpr IndexRangeTuple coords2(IRng{1, 4}, ALL);
constexpr auto coords2_range_start = get_index_range_start(coords2);
static_assert(coords2_range_start.size() == 2);
static_assert(coords2_range_start[0] == 1);
static_assert(coords2_range_start[1] == 0);
}  // namespace get_index_range_start_tests

namespace is_index_range_empty_tests
{
static_assert(is_index_range_empty(IndexRangeTuple{}));
static_assert(is_index_range_empty(IndexRangeTuple{IRng{}}));
static_assert(is_index_range_empty(IndexRangeTuple{IRng{0, 2}, IRng{}}));
static_assert(!is_index_range_empty(IndexRangeTuple{IRng{0}}));
static_assert(!is_index_range_empty(IndexRangeTuple{IRng{0, 2}, ALL}));
static_assert(!is_index_range_empty(IndexRangeTuple{ALL}));
}  // namespace is_index_range_empty_tests

namespace get_index_range_shape_tests
{
constexpr ShapeTuple empty_shape;
constexpr IndexRangeTuple empty_coord;
constexpr auto empty_range_shape = get_index_range_shape(empty_coord, empty_shape);
static_assert(empty_range_shape.size() == 0);

constexpr ShapeTuple shape1(8, 8);
constexpr IndexRangeTuple coord1(IRng(0, 2), IRng(0, 3));
constexpr auto range_shape1 = get_index_range_shape(coord1, shape1);
static_assert(range_shape1.size() == 2);
static_assert(range_shape1[0] == 2);
static_assert(range_shape1[1] == 3);

constexpr IndexRangeTuple coord2(IRng(0, 2), ALL);
constexpr auto range_shape2 = get_index_range_shape(coord2, shape1);
static_assert(range_shape2.size() == 2);
static_assert(range_shape2[0] == 2);
static_assert(range_shape2[1] == 8);

constexpr IndexRangeTuple coord3(ALL);
constexpr auto range_shape3 = get_index_range_shape(coord3, shape1);
static_assert(range_shape3.size() == 2);
static_assert(range_shape3[0] == 8);
static_assert(range_shape3[1] == 8);

constexpr IndexRangeTuple coord4(IRng(0));
constexpr auto range_shape4 = get_index_range_shape(coord4, shape1);
static_assert(range_shape4.size() == 1);
static_assert(range_shape4[0] == 8);

constexpr IndexRangeTuple coord5(IRng(0), IRng(2));
constexpr auto range_shape5 = get_index_range_shape(coord5, shape1);
static_assert(range_shape5.size() == 0);

constexpr IndexRangeTuple coord6(IRng(0), IRng(2, 4));
constexpr auto range_shape6 = get_index_range_shape(coord6, shape1);
static_assert(range_shape6.size() == 1);
static_assert(range_shape6[0] == 2);
}  // get_index_range_shape_tests

namespace is_index_range_contained_tests
{
constexpr ShapeTuple shape(3, 2, 1);

constexpr IndexRangeTuple coord1(ALL, ALL, ALL);
static_assert(is_index_range_contained(coord1, shape));

constexpr IndexRangeTuple coord2(IRng(0, 3), IRng(0, 2), IRng(0, 1));
static_assert(is_index_range_contained(coord2, shape));

constexpr IndexRangeTuple coord3(IRng(0, 4), ALL, ALL);
static_assert(!is_index_range_contained(coord3, shape));

constexpr IndexRangeTuple coord4(ALL, ALL, IRng(1, 2));
static_assert(!is_index_range_contained(coord4, shape));

constexpr IndexRangeTuple coord5(IRng(1, 3), ALL);
static_assert(is_index_range_contained(coord5, shape));
}  // namespace is_index_range_contained_tests

namespace do_index_ranges_intersect_tests
{
static_assert(do_index_ranges_intersect(IRng(0, 1), IRng(0, 2)));
static_assert(do_index_ranges_intersect(IRng(3, 5), IRng(0, 4)));
static_assert(do_index_ranges_intersect(IRng(0, 1), IRng(0, 1)));
static_assert(!do_index_ranges_intersect(IRng(0, 2), IRng(2, 4)));
static_assert(do_index_ranges_intersect(IRng(0, 1), ALL));
static_assert(do_index_ranges_intersect(ALL, ALL));
static_assert(!do_index_ranges_intersect(IRng(), IRng(0, 4)));

static_assert(do_index_ranges_intersect(IndexRangeTuple(IRng(0, 1), IRng(0, 2)),
                                        IndexRangeTuple(IRng(0, 2),
                                                        IRng(0, 1))));
static_assert(
    !do_index_ranges_intersect(IndexRangeTuple(IRng(0, 1), IRng(0, 2)),
                               IndexRangeTuple(IRng(1, 2), IRng(0, 2))));
static_assert(do_index_ranges_intersect(IndexRangeTuple(IRng(0, 1), IRng(0, 2)),
                                        IndexRangeTuple(IRng(0, 1), ALL)));
}  // namespace do_index_ranges_intersect_tests

namespace intersect_index_ranges_tests
{
static_assert(intersect_index_ranges(IRng(0, 1), IRng(0, 1)) == IRng(0, 1));
static_assert(intersect_index_ranges(IRng(0, 1), IRng(0, 2)) == IRng(0, 1));
static_assert(intersect_index_ranges(IRng(0, 2), IRng(1, 3)) == IRng(1, 2));
static_assert(intersect_index_ranges(IRng(0, 2), ALL) == IRng(0, 2));
static_assert(intersect_index_ranges(ALL, ALL) == ALL);

static_assert(intersect_index_ranges(IndexRangeTuple(IRng(0, 1), IRng(0, 2)),
                                     IndexRangeTuple(IRng(0, 2), IRng(0, 1)))
              == IndexRangeTuple(IRng(0, 1), IRng(0, 1)));
static_assert(intersect_index_ranges(IndexRangeTuple(IRng(0, 1), IRng(0, 2)),
                                     IndexRangeTuple(IRng(0, 1), ALL))
              == IndexRangeTuple(IRng(0, 1), IRng(0, 2)));
}  // namespace intersect_index_ranges_tests

namespace is_index_in_shape_tests
{
static_assert(is_index_in_shape(ScalarIndexTuple{}, ShapeTuple{}));
static_assert(is_index_in_shape(ScalarIndexTuple{0, 0}, ShapeTuple{2, 2}));
static_assert(!is_index_in_shape(ScalarIndexTuple{2, 2}, ShapeTuple{2, 2}));
}  // namespace is_index_in_shape_tests

namespace next_scalar_index_tests
{
static_assert(next_scalar_index({0}, {2}) == ScalarIndexTuple{1});
static_assert(next_scalar_index({0, 0}, {2, 2}) == ScalarIndexTuple{1, 0});
static_assert(next_scalar_index({1, 0}, {2, 2}) == ScalarIndexTuple{0, 1});
static_assert(next_scalar_index({1}, {2}) == ScalarIndexTuple{2});
}  // namespace next_scalar_index_tests
