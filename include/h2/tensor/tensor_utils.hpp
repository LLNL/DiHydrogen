////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/tensor/tensor_types.hpp"

/** @file
 *
 * Utilities for working with tensors.
 */

namespace h2
{

/**
 * Return the coordinates of the initial point of coordinate range.
 */
constexpr inline SingleCoordTuple get_range_start(CoordTuple coords) H2_NOEXCEPT {
  SingleCoordTuple coords_start(TuplePad<SingleCoordTuple>(coords.size()));
  for (typename CoordTuple::size_type i = 0; i < coords.size(); ++i) {
    // Abuse the fact that ALL.start = 0.
    coords_start[i] = coords[i].start;
  }
  return coords_start;
}

/**
 * Return true if the DimensionRange is trivial.
 */
constexpr inline bool is_coord_trivial(DimensionRange coord) H2_NOEXCEPT {
  return coord.start + 1 == coord.end;
}

/**
 * Return true if the DimensionRange is empty (i.e., start == end).
 */
constexpr inline bool is_coord_empty(DimensionRange coord) H2_NOEXCEPT
{
  return coord.start == coord.end;
}

/**
 * Return true if the coordinate range is empty.
 *
 * This occurs when at least one entry in the range is empty or the
 * range itself is empty.
 */
constexpr inline bool is_range_empty(CoordTuple coords) H2_NOEXCEPT
{
  return coords.is_empty() || any_of(coords, is_coord_empty);
}

/**
 * Return the shape defined by a coordinate range within a larger shape,
 * eliminating trivial dimensions.
 */
constexpr inline ShapeTuple get_range_shape(CoordTuple coords, ShapeTuple shape) H2_NOEXCEPT {
  H2_ASSERT_DEBUG(coords.size() <= shape.size(),
                  "coords size not compatible with shape size");
  ShapeTuple new_shape(TuplePad<ShapeTuple>(shape.size()));
  typename ShapeTuple::size_type j = 0;
  for (typename ShapeTuple::size_type i = 0; i < shape.size(); ++i) {
    if (i >= coords.size() || coords[i] == ALL) {
      new_shape[j] = shape[i];
      ++j;
    } else if (!is_coord_trivial(coords[i])) {
      new_shape[j] = coords[i].end - coords[i].start;
      ++j;
    }
  }
  new_shape.set_size(j);
  return new_shape;
}

/**
 * Return true if a coordinate range is contained within a given shape.
 */
constexpr inline bool is_shape_contained(CoordTuple coords,
                                         ShapeTuple shape) H2_NOEXCEPT
{
  if (coords.size() > shape.size())
  {
    return false;
  }
  for (typename CoordTuple::size_type i = 0; i < coords.size(); ++i)
  {
    if (coords[i] != ALL
        && (coords[i].start > shape[i] || coords[i].end > shape[i]))
    {
      return false;
    }
  }
  return true;
}

/**
 * Return a new tuple that consists of the entries in the original at
 * indices where coords is not trivial.
 */
template <typename T, typename SizeType, SizeType N>
constexpr inline FixedSizeTuple<T, SizeType, N> filter_by_trivial(
  CoordTuple coords, FixedSizeTuple<T, SizeType, N> tuple) H2_NOEXCEPT {
  FixedSizeTuple<T, SizeType, N> new_tuple(
    FixedSizeTuplePadding<T, SizeType>(tuple.size(), T{}));
  SizeType j = 0;
  for (SizeType i = 0; i < tuple.size(); ++i) {
    if (i >= coords.size() || !is_coord_trivial(coords[i])) {
      new_tuple[j] = tuple[i];
      ++j;
    }
  }
  new_tuple.set_size(j);
  return new_tuple;
}

/**
 * Iterate over an n-dimensional region.
 *
 * The given function f will be called with a `SingleCoordTuple` for
 * each coordinate position.
 *
 * @todo In the future, we could specialize for specific dimensions.
 */
template <typename Func>
void for_ndim(ShapeTuple shape, Func f)
{
  if (shape.is_empty())
  {
    return;
  }
  SingleCoordTuple coord(TuplePad<SingleCoordTuple>(shape.size(), 0));
  const DataIndexType ub = product<DataIndexType>(shape);
  for (DataIndexType i = 0; i < ub; ++i)
  {
    f(coord);
    coord[0] += 1;
    for (typename SingleCoordTuple::size_type dim = 0; dim < coord.size() - 1;
         ++dim)
    {
      if (coord[dim] == shape[dim])
      {
        coord[dim] = 0;
        coord[dim + 1] += 1;
      }
    }
  }
}

}  // namespace h2
