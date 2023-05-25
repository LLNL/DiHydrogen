////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <cstdint>

#include <El.hpp>

#include "h2/tensor/fixed_size_tuple.hpp"

/** @file
 *
 * Various types and helpers for defining tensors.
 */

namespace h2 {

/**
 * Indicates the type of a tensor dimension.
 *
 * A tensor dimension type is a tag that indicates the semantics of a
 * given dimension. This is used to construct mappings to certain
 * algorithms which associate particular semantics to dimensions (e.g.,
 * the sample dimension).
 *
 * These are not used for other sorts of correctness checking.
 */
enum class DimensionType {
  Any,  /**< Catch-all, does not ascribe particular semantics. */
  Sample,  /**< The sample ("batch") dimension. */
  Channel,  /**< The channel ("feature") dimension in convolutions. */
  Filter,  /**< The filter dimension in convolutions. */
  Spatial,  /**< The spatial (height, width, depth, etc.) dimension(s). */
  Sequence  /**< The sequence dimension (e.g., in textual data). */
};

using DT = DimensionType;  // Alias to save you some typing.

/**
 * Compute device type (e.g., CPU, GPU).
 */
using Device = El::Device;  // Leverage Hydrogen's device typing.

/**
 * Integer type used for the number of dimensions.
 */
using NDimType = std::int32_t;

/**
 * Integer type used for storing dimensions.
 */
using DimType = std::int32_t;

/**
 * Integer type used for data indices.
 */
using DataIndexType = std::int64_t;

/**
 * Maximum number of dimensions a tensor may have.
 */
static constexpr NDimType MAX_TENSOR_DIMS = 8;

/**
 * Fixed-size tuple where the SizeType is NDimType and the max size is
 * MAX_TENSOR_DIMS.
 */
template <typename T>
using NDimTuple = FixedSizeTuple<T, NDimType, MAX_TENSOR_DIMS>;

/**
 * The shape of a tensor, a tuple with some number of integral values.
 */
using ShapeTuple = NDimTuple<DimType>;

/**
 * Tuple of dimension types.
 */
using DimensionTypeTuple = NDimTuple<DimensionType>;

using DTTuple = DimensionTypeTuple;  // Alias to save you some typing.

/**
 * The strides of a tensor.
 */
using StrideTuple = NDimTuple<DataIndexType>;

/**
 * Represents a contiguous range [start, end).
 */
struct DimensionRange {
  DimType start;  /**< Start of a range. */
  DimType end;  /**< End of a range. */
  constexpr DimensionRange() : start(0), end(0) {}
  constexpr DimensionRange(DimType i) : start(i), end(i+1) {}
  constexpr DimensionRange(DimType start_, DimType end_) : start(start_), end(end_) {}

  constexpr bool operator==(const DimensionRange& other) H2_NOEXCEPT {
    return start == other.start && end == other.end;
  }
};

using DRng = DimensionRange;  // Alias to save you some typing.

/**
 * Tuple of dimension ranges.
 */
using CoordTuple = NDimTuple<DimensionRange>;

/** Special DimensionRange that represents a entire range. */
static constexpr DimensionRange ALL(0, -1);

/**
 * Tuple of exact coordinates.
 */
using SingleCoordTuple = NDimTuple<DimType>;

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
 * Specifies the type of view.
 */
enum class ViewType {
  None,  /**< Not a view. */
  Mutable,  /**< A view that can modify the original. */
  Const  /**< A view that cannot modify the original. */
};

}  // namespace h2
