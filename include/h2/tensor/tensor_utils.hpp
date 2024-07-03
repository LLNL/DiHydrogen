////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/tensor/tensor_types.hpp"
#include "tensor_types.hpp"

/** @file
 *
 * Utilities for working with tensors.
 */

namespace h2
{

/**
 * Convert a `ScalarIndexTuple` to a corresponding `IndexRangeTuple`.
 */
constexpr inline IndexRangeTuple
scalar2range_tuple(const ScalarIndexTuple& tuple) H2_NOEXCEPT
{
  IndexRangeTuple ir_tuple(TuplePad<IndexRangeTuple>(tuple.size()));
  for (ScalarIndexTuple::size_type i = 0; i < tuple.size(); ++i)
  {
    ir_tuple[i] = IndexRange(tuple[i]);
  }
  return ir_tuple;
}

/**
 * Return a scalar index tuple denoting the start of an index range.
 *
 * This is the starting point of each index range in the tuple.
 */
constexpr inline ScalarIndexTuple
get_index_range_start(const IndexRangeTuple& coords) H2_NOEXCEPT
{
  ScalarIndexTuple coords_start(TuplePad<ScalarIndexTuple>(coords.size()));
  for (typename IndexRangeTuple::size_type i = 0; i < coords.size(); ++i) {
    // No special case for ALL, that starts at 0.
    coords_start[i] = coords[i].start();
  }
  return coords_start;
}

/**
 * Return true if the index range is empty.
 *
 * This occurs when at least one entry in the range is empty or the
 * range itself is empty.
 */
constexpr inline bool
is_index_range_empty(const IndexRangeTuple& coords) H2_NOEXCEPT
{
  return coords.is_empty()
         || any_of(coords, [](const typename IndexRangeTuple::type& c) {
              return c.is_empty();
            });
}

/**
 * Return the shape defined by an index range within a larger shape,
 * eliminating scalar dimensions.
 *
 * If any index ranges in `coords` are empty, the behavior of this is
 * undefined. (However, `coords` itself may be empty, which yields an
 * empty shape.)
 */
constexpr inline ShapeTuple
get_index_range_shape(const IndexRangeTuple& coords,
                      const ShapeTuple& shape) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(coords.size() <= shape.size(),
                  "coords size (",
                  coords,
                  ") not compatible with shape size (",
                  shape,
                  ")");
  H2_ASSERT_DEBUG(!is_index_range_empty(coords) || coords.is_empty(),
                  "get_index_range_shape does not work with empty ranges");
  ShapeTuple new_shape(TuplePad<ShapeTuple>(shape.size()));
  typename ShapeTuple::size_type j = 0;
  for (typename ShapeTuple::size_type i = 0; i < shape.size(); ++i) {
    if (i >= coords.size() || coords[i] == ALL) {
      new_shape[j] = shape[i];
      ++j;
    } else if (!coords[i].is_scalar()) {
      new_shape[j] = coords[i].end() - coords[i].start();
      ++j;
    }
  }
  new_shape.set_size(j);
  return new_shape;
}

/**
 * Return true if an index range is contained within a given shape.
 */
constexpr inline bool
is_index_range_contained(const IndexRangeTuple& coords,
                         const ShapeTuple& shape) H2_NOEXCEPT
{
  if (coords.size() > shape.size())
  {
    return false;
  }
  for (typename IndexRangeTuple::size_type i = 0; i < coords.size(); ++i)
  {
    if (coords[i] != ALL
        && (coords[i].start() > shape[i] || coords[i].end() > shape[i]))
    {
      return false;
    }
  }
  return true;
}

/**
 * Return true if two index ranges have a non-empty intersection.
 *
 * The index ranges may not be scalar.
 */
constexpr inline bool
do_index_ranges_intersect(const IndexRange& ir1,
                          const IndexRange& ir2) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(!ir1.is_scalar() && !ir2.is_scalar(),
                  "Cannot intersect scalar index ranges ",
                  ir1,
                  " and ",
                  ir2);
  return !ir1.is_empty() && !ir2.is_empty()
         && ((ir1 == ALL) || (ir2 == ALL)
             || (ir1.start() < ir2.end() && ir2.start() < ir1.end()));
}

/**
 * Return true if two index ranges have a non-empty intersection.
 *
 * The index ranges may not have scalar entries.
 */
constexpr inline bool
do_index_ranges_intersect(const IndexRangeTuple& ir1,
                          const IndexRangeTuple& ir2) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(ir1.size() == ir2.size(),
                  "Index ranges ",
                  ir1,
                  " and ",
                  ir2,
                  " must be the same size to intersect");
  for (typename IndexRangeTuple::size_type i = 0; i < ir1.size(); ++i)
  {
    if (!do_index_ranges_intersect(ir1[i], ir2[i]))
    {
      return false;
    }
  }
  return true;
}

/**
 * Return the intersection of of two index ranges.
 *
 * The index ranges must have a non-empty intersection.
 */
constexpr inline IndexRange
intersect_index_ranges(const IndexRange& ir1, const IndexRange& ir2) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(do_index_ranges_intersect(ir1, ir2),
                  "Index ranges ", ir1, " and ", ir2, " must intersect");
  return IndexRange(std::max(ir1.start(), ir2.start()),
                    std::min(ir1.end(), ir2.end()));
}

/**
 * Return the intersection of of two index ranges.
 *
 * The index ranges must have a non-empty intersection.
 */
constexpr inline IndexRangeTuple
intersect_index_ranges(const IndexRangeTuple& ir1,
                       const IndexRangeTuple& ir2) H2_NOEXCEPT
{
  H2_ASSERT_DEBUG(ir1.size() == ir2.size(),
                  "Index ranges ",
                  ir1,
                  " and ",
                  ir2,
                  " must be the same size to intersect");
  H2_ASSERT_DEBUG(do_index_ranges_intersect(ir1, ir2),
                  "Index ranges ",
                  ir1,
                  " and ",
                  ir2,
                  " must intersect");
  return map_index(ir1, [&ir1, &ir2](IndexRangeTuple::size_type i) {
    return intersect_index_ranges(ir1[i], ir2[i]);
  });
}

/**
 * Iterate over an n-dimensional region.
 *
 * The given function f will be called with a `ScalarIndexTuple` for
 * each index position.
 *
 * @todo In the future, we could specialize for specific dimensions.
 */
template <typename Func>
void for_ndim(const ShapeTuple& shape, Func f)
{
  if (shape.is_empty())
  {
    return;
  }
  ScalarIndexTuple coord(TuplePad<ScalarIndexTuple>(shape.size(), 0));
  const DataIndexType ub = product<DataIndexType>(shape);
  for (DataIndexType i = 0; i < ub; ++i)
  {
    f(coord);
    coord[0] += 1;
    for (typename ScalarIndexTuple::size_type dim = 0; dim < coord.size() - 1;
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
