////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <ostream>
#include <type_traits>

#include <El.hpp>

#include "h2/tensor/fixed_size_tuple.hpp"
#include "h2/tensor/tuple_utils.hpp"
#include "h2/core/device.hpp"
#include "h2/core/sync.hpp"

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
enum class DimensionType
{
  Any,     /**< Catch-all, does not ascribe particular semantics. */
  Scalar,  /** Internal use for views that eliminate all dimension. */
  Sample,  /**< The sample ("batch") dimension. */
  Channel, /**< The channel ("feature") dimension in convolutions. */
  Filter,  /**< The filter dimension in convolutions. */
  Spatial, /**< The spatial (height, width, depth, etc.) dimension(s). */
  Sequence /**< The sequence dimension (e.g., in textual data). */
};

using DT = DimensionType;  // Alias to save you some typing.

/** Support printing DimensionType. */
inline std::ostream& operator<<(std::ostream& os, const DimensionType& dim_type)
{
  switch (dim_type)
  {
  case DT::Any:
    os << "Any";
    break;
  case DT::Scalar:
    os << "Scalar";
    break;
  case DT::Sample:
    os << "Sample";
    break;
  case DT::Channel:
    os << "Channel";
    break;
  case DT::Filter:
    os << "Filter";
    break;
  case DT::Spatial:
    os << "Spatial";
    break;
  case DT::Sequence:
    os << "Sequence";
    break;
  default:
    os << "Unknown";
    break;
  }
  return os;
}

/**
 * Integer type used for the number of dimensions.
 */
using NDimType = std::int32_t;

/**
 * Integer type used for storing the size of a dimension.
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
 * Represents a range of indices.
 *
 * This is either a single scalar index or a half-open range containing
 * [start, stop).
 *
 * If the range is a scalar, both start and end will have the same
 * value.
 *
 * The range may also be empty, in which case the values of `start` and
 * `stop` are undefined.
 */
struct IndexRange
{
  /** Construct an empty IndexRange. */
  constexpr IndexRange() : index_start(0), index_end(-1) {}
  /** Construct a scalar IndexRange. */
  constexpr IndexRange(DimType i) : index_start(i), index_end(i) {}
  /** Construct a half-open IndexRange. */
  constexpr IndexRange(DimType start_, DimType end_)
      : index_start(start_),
        index_end(end_)
  {
    H2_ASSERT_DEBUG(start_ < end_,
                    "IndexRange with end <= start not supported, you probably "
                    "have a bug or want an empty IndexRange");
  }

  constexpr inline DimType start() const H2_NOEXCEPT { return index_start; }
  constexpr inline DimType end() const H2_NOEXCEPT { return index_end; }
  constexpr inline bool is_scalar() const H2_NOEXCEPT
  {
    return index_start == index_end;
  }
  constexpr inline bool is_empty() const H2_NOEXCEPT
  {
    return index_end < index_start;
  }

private:
  // Implementation detail: index_end < index_start is used to denote
  // an empty range. To prevent "gotchas" if DimType changes, enforce
  // that is be signed so this holds.
  static_assert(std::is_signed_v<DimType>,
                "Underlying dimension type for IndexRange must be signed");

  DimType index_start;  /**< Start of a range. */
  DimType index_end;    /**< End of a range. */
};

/** Equality for ranges. */
inline constexpr bool operator==(const IndexRange& ir1,
                                 const IndexRange& ir2)
{
  return ir1.start() == ir2.start() && ir1.end() == ir2.end();
}

/** Inequality for ranges. */
inline constexpr bool operator!=(const IndexRange& ir1,
                                 const IndexRange& ir2)
{
  return ir1.start() != ir2.start() || ir1.end() != ir2.end();
}

using IRng = IndexRange;  // Alias to save you some typing.

/** Special IndexRange that represents a entire range. */
static constexpr IndexRange ALL(0, std::numeric_limits<DimType>::max());

/** Support printing IndexRange. */
inline std::ostream& operator<<(std::ostream& os, const IndexRange& ir)
{
  if (ir == ALL)
  {
    os << "[ALL]";
  }
  else if (ir.is_scalar())
  {
    os << "[" << ir.start() << "]";
  }
  else if (ir.is_empty())
  {
    os << "[empty]";
  }
  else
  {
    os << "[" << ir.start() << ", " << ir.end() << ")";
  }
  return os;
}

/**
 * Tuple of IndexRanges, which represent a region.
 */
using IndexRangeTuple = NDimTuple<IndexRange>;

/**
 * Tuple of scalar indices, which represent a point.
 */
using ScalarIndexTuple = NDimTuple<DimType>;

/**
 * Specifies the type of view.
 */
enum class ViewType
{
  None,    /**< Not a view. */
  Mutable, /**< A view that can modify the original. */
  Const    /**< A view that cannot modify the original. */
};

/** Support printing ViewType. */
inline std::ostream& operator<<(std::ostream& os, const ViewType& vt)
{
  switch (vt)
  {
  case ViewType::None:
    os << "None";
    break;
  case ViewType::Mutable:
    os << "View";
    break;
  case ViewType::Const:
    os << "Const View";
    break;
  default:
    os << "Unknown";
  }
  return os;
}

// These are used by local and distributed tensors for memory recovery.
/** Do not attempt recovery in `BaseTensor::ensure`. */
static constexpr struct tensor_no_recovery_t {} TensorNoRecovery;
/** Attempt recovery in `BaseTensor::ensure`. */
static constexpr struct tensor_attempt_recovery_t {} TensorAttemptRecovery;

/** Tag to indicate a tensor should allocate lazily. */
static constexpr struct lazy_alloc_t {} LazyAlloc;
/** Tag to indicate a tensor should not allocate lazily. */
static constexpr struct strict_alloc_t {} StrictAlloc;

}  // namespace h2
