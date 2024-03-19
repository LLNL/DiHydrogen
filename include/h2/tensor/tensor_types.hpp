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

#include <El.hpp>

#include "h2/tensor/fixed_size_tuple.hpp"
#include "h2/tensor/tuple_utils.hpp"

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

/** Support printing DimensionType. */
inline std::ostream& operator<<(std::ostream& os, const DimensionType& dim_type)
{
  switch (dim_type)
  {
  case DT::Any:
    os << "Any";
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
 * Compute device type (e.g., CPU, GPU).
 */
using Device = El::Device;  // Leverage Hydrogen's device typing.

/** Support printing Device. */
inline std::ostream& operator<<(std::ostream& os, const Device& dev)
{
  switch (dev)
  {
  case Device::CPU:
    os << "CPU";
    break;
#ifdef H2_HAS_GPU
  case Device::GPU:
    os << "GPU";
    break;
#endif
  default:
    os << "Unknown";
    break;
  }
  return os;
}

/**
 * Manage device-specific synchronization.
 *
 * A note on how Tensors deal with synchronization (here because there
 * isn't a great place to write this since it touches many classes):
 *
 * When a Tensor is created, the creator may either specify the
 * SyncInfo for the Tensor to use, or the Tensor will create one with
 * the default SyncInfo constructor for the appropriate Device. This
 * SyncInfo is passed through to the underlying StridedMemory and
 * RawBuffer. The RawBuffer will allocate any memory using that
 * SyncInfo. Any Tensor operation that changes the underlying RawBuffer
 * (e.g., `empty`, `resize`) will continue to use the SyncInfo
 * associated with the Tensor. As a special case of this, an empty
 * Tensor, which has no RawBuffer, will use the Tensor's SyncInfo
 * should it construct a RawBuffer (e.g., due to being resized).
 *
 * When a view of a Tensor is created, the viewing Tensor will default
 * to the same SyncInfo as the original Tensor.
 *
 * When a Tensor wraps external memory (by providing a raw pointer),
 * there is again no RawBuffer created and the Tensor's SyncInfo will
 * be used for all operations.
 *
 * The get/set_sync_info methods may be used on Tensors and RawBuffers
 * to retrieve or change the associated SyncInfo. get_sync_info on a
 * Tensor always returns the Tensor's SyncInfo, which may be different
 * from the SyncInfo associated with the RawBuffer underlying the
 * Tensor (due to set_sync_info).
 *
 * If the SyncInfo on a Tensor is changed (via set_sync_info), the
 * semantics depend on whether the Tensor is a view. If the Tensor is
 * not a view, this will also change the SyncInfo of the underlying
 * RawBuffer. If the Tensor is a view, only the Tensor's SyncInfo will
 * be changed. (This is how a Tensor's SyncInfo may differ from its
 * RawBuffer's.) This enables views of the same Tensor to enqueue
 * operations on multiple compute streams concurrently; it is up to the
 * user to ensure the appropriate synchronization in such uses.
 *
 * This requires careful handling of destruction in the RawBuffer, as
 * there may be operations on multiple compute streams accessing the
 * data, yet the RawBuffer is only (directly) associated with one
 * SyncInfo. In particular, consider the case where an initial Tensor A
 * is created with SyncInfo SA, and then a view, B, of that Tensor is
 * created and associated with SyncInfo SB. The underlying RawBuffer
 * will be associated with SA. If A is deleted, the RawBuffer will
 * still exist, as B still has a reference to it. Now suppose B
 * launches some operations on SB, then is itself deleted. The
 * operations should continue to run fine, due to the compute stream
 * ordering. However, the RawBuffer's deletion will be synchronized
 * only to SA, potentially leading to a race with operations on SB. To
 * avoid this, whenever a Tensor discards a reference to a RawBuffer,
 * it informs the RawBuffer it is doing so, along with its current
 * SyncInfo. If the SyncInfo differs from the RawBuffer's, it will
 * record an event on the Tensor's SyncInfo and keep a reference to it.
 * When the RawBuffer is deleted, it will synchronize with all recorded
 * SyncInfos before enqueuing the delete, to avoid races.
 *
 * Another situation to be aware of: If you change a Tensor's SyncInfo,
 * it is up to you to provide any needed synchronization between the
 * original SyncInfo and the new one.
 *
 * An implementation note (separate from the above semantics): SyncInfo
 * objects are stored in StridedMemory, rather than directly in a
 * Tensor. This is just to simplify implementation.
 */
template <Device Dev>
using SyncInfo = El::SyncInfo<Dev>;

/** Support printing SyncInfo. */
template <Device Dev>
inline std::ostream& operator<<(std::ostream&, const SyncInfo<Dev>& sync);

template <>
inline std::ostream& operator<<(std::ostream& os,
                                const SyncInfo<Device::CPU>& sync)
{
  os << "CPUSync";
  return os;
}

#ifdef H2_HAS_GPU
template <>
inline std::ostream& operator<<(std::ostream& os,
                                const SyncInfo<Device::GPU>& sync)
{
  os << "GPUSync(" << sync.Stream() << ")";
  return os;
}
#endif

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
};

/** Equality for ranges. */
inline constexpr bool operator==(const DimensionRange& dr1,
                                 const DimensionRange& dr2)
{
  return dr1.start == dr2.start && dr1.end == dr2.end;
}

/** Inequality for ranges. */
inline constexpr bool operator!=(const DimensionRange& dr1,
                                 const DimensionRange& dr2)
{
  return dr1.start != dr2.start || dr1.end != dr2.end;
}

using DRng = DimensionRange;  // Alias to save you some typing.

/** Special DimensionRange that represents a entire range. */
static constexpr DimensionRange ALL(0, std::numeric_limits<DimType>::max());

/** Support printing DimensionRange. */
inline std::ostream& operator<<(std::ostream& os, const DimensionRange& dr)
{
  if (dr == ALL)
  {
    os << "[ALL]";
  }
  else
  {
    os << "[" << dr.start << ", " << dr.end << ")";
  }
  return os;
}

/**
 * Tuple of dimension ranges.
 */
using CoordTuple = NDimTuple<DimensionRange>;

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
 * Specifies the type of view.
 */
enum class ViewType
{
  None,    /**< Not a view. */
  Mutable, /**< A view that can modify the original. */
  Const    /**< A view that cannot modify the original. */
};

// These are used by local and distributed tensors for memory recovery.
/** Do not attempt recovery in `BaseTensor::ensure`. */
static constexpr struct tensor_no_recovery_t {} TensorNoRecovery;
/** Attempt recovery in `BaseTensor::ensure`. */
static constexpr struct tensor_attempt_recovery_t {} TensorAttemptRecovery;

/** Tag to indicate a tensor should allocate lazily. */
static constexpr struct lazy_alloc_t {} LazyAlloc;
/** Tag to indicate a tensor should not allocate lazily. */
static constexpr struct unlazy_alloc_t {} UnlazyAlloc;

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
