////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/tensor_types.hpp"

using namespace h2;

namespace is_coord_trivial_tests {
constexpr DimensionRange triv_dimrange(0, 1);
static_assert(is_coord_trivial(triv_dimrange),
              "Trivial coord is not trivial");

constexpr DimensionRange nontriv_dimrange(0, 2);
static_assert(!is_coord_trivial(nontriv_dimrange),
              "Non-trivial coord is trivial");

static_assert(!is_coord_trivial(ALL),
              "ALL is trivial");

constexpr DimensionRange triv_dimrange_single(0);
static_assert(is_coord_trivial(triv_dimrange_single),
              "Trivial coord is not trivial");

constexpr DimensionRange default_dimrange;
static_assert(!is_coord_trivial(default_dimrange),
              "Default DimensionRange is trivial");
}

namespace get_range_start_tests {
constexpr CoordTuple empty_coords;
static_assert(get_range_start(empty_coords).size() == 0,
              "get_range_start of empty coords is not empty");

constexpr CoordTuple coords1(1, 2);
constexpr auto coords1_range_start = get_range_start(coords1);
static_assert(coords1_range_start.size() == 2,
              "get_range_start returned wrong size");
static_assert(coords1_range_start[0] == 1,
              "get_range_start returned wrong start");
static_assert(coords1_range_start[1] == 2,
              "get_range_start returned wrong start");

constexpr CoordTuple coords2(DRng{1, 4}, ALL);
constexpr auto coords2_range_start = get_range_start(coords2);
static_assert(coords2_range_start.size() == 2,
              "get_range_start returned wrong size");
static_assert(coords2_range_start[0] == 1,
              "get_range_start returned wrong start");
static_assert(coords2_range_start[1] == 0,
              "get_range_start returned wrong start");
}

namespace get_range_shape_tests {
constexpr ShapeTuple empty_shape;
constexpr CoordTuple empty_coord;
constexpr auto empty_range_shape = get_range_shape(empty_coord, empty_shape);
static_assert(empty_range_shape.size() == 0,
              "get_range_shape not empty");

constexpr ShapeTuple shape1(8, 8);
constexpr CoordTuple coord1(DRng(0, 2), DRng(0, 3));
constexpr auto range_shape1 = get_range_shape(coord1, shape1);
static_assert(range_shape1.size() == 2,
              "get_range_shape returned wrong size");
static_assert(range_shape1[0] == 2,
              "get_range_shape index 0 is wrong");
static_assert(range_shape1[1] == 3,
              "get_range_shape index 1 is wrong");

constexpr CoordTuple coord2(DRng(0, 2), ALL);
constexpr auto range_shape2 = get_range_shape(coord2, shape1);
static_assert(range_shape2.size() == 2,
              "get_range_shape returned wrong size");
static_assert(range_shape2[0] == 2,
              "get_range_shape index 0 is wrong");
static_assert(range_shape2[1] == 8,
              "get_range_shape index 1 is wrong");

constexpr CoordTuple coord3(ALL);
constexpr auto range_shape3 = get_range_shape(coord3, shape1);
static_assert(range_shape3.size() == 2,
              "get_range_shape returned wrong size");
static_assert(range_shape3[0] == 8,
              "get_range_shape index 0 is wrong");
static_assert(range_shape3[1] == 8,
              "get_range_shape index 1 is wrong");

constexpr CoordTuple coord4(DRng(0));
constexpr auto range_shape4 = get_range_shape(coord4, shape1);
static_assert(range_shape4.size() == 1,
              "get_range_shape returned wrong size");
static_assert(range_shape4[0] == 8,
              "get_range_shape index 0 is wrong");

constexpr CoordTuple coord5(DRng(0), DRng(2));
constexpr auto range_shape5 = get_range_shape(coord5, shape1);
static_assert(range_shape5.size() == 0,
              "get_range_shape returned wrong size");

constexpr CoordTuple coord6(DRng(0), DRng(2, 4));
constexpr auto range_shape6 = get_range_shape(coord6, shape1);
static_assert(range_shape6.size() == 1,
              "get_range_shape returned wrong size");
static_assert(range_shape6[0] == 2,
              "get_range_shape index 0 is wrong");
}

namespace filter_by_trivial_tests {
constexpr ShapeTuple tuple(1, 2, 3);

constexpr CoordTuple coord1;
constexpr auto filtered_tuple1 = filter_by_trivial(coord1, tuple);
static_assert(filtered_tuple1.size() == 3,
              "filter_by_trivial returned wrong size");
static_assert(filtered_tuple1[0] == 1,
              "filter_by_trivial index 0 is wrong");
static_assert(filtered_tuple1[1] == 2,
              "filter_by_trivial index 1 is wrong");
static_assert(filtered_tuple1[2] == 3,
              "filter_by_trivial index 2 is wrong");

constexpr CoordTuple coord2(DRng(0), DRng(1), DRng(2));
constexpr auto filtered_tuple2 = filter_by_trivial(coord2, tuple);
static_assert(filtered_tuple2.size() == 0,
              "filter_by_trivial returned wrong size");

constexpr CoordTuple coord3(ALL, DRng(0));
constexpr auto filtered_tuple3 = filter_by_trivial(coord3, tuple);
static_assert(filtered_tuple3.size() == 2,
              "filter_by_trivial returned wrong size");
static_assert(filtered_tuple3[0] == 1,
              "filter_by_trivial index 0 is wrong");
static_assert(filtered_tuple3[1] == 3,
              "filter_by_trivial index 1 is wrong");
}
