////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * This defines the public API for local (non-distributed) tensors.
 */

#include <h2_config.hpp>

#include <array>
#include <cstdint>

#include <El.hpp>

#include "h2/utils/Error.hpp"

namespace h2
{

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

/** Represent a tuple with a constant maximum size. */
template <typename T, NDimType N>
struct FixedSizeTuple {
  std::array<T, N> data_;  /**< Fixed size data buffer. */
  NDimType size_; /**< Number of valid elements in the tuple. */

  template <typename... Args>
  constexpr FixedSizeTuple(Args... args) : data_{args...}, size_(sizeof...(args)) {}

  FixedSizeTuple(const FixedSizeTuple& other) = default;
  FixedSizeTuple(FixedSizeTuple&& other) = default;
  FixedSizeTuple& operator=(const FixedSizeTuple& other) = default;
  FixedSizeTuple& operator=(FixedSizeTuple&& other) = default;

  /** Return the number of valid elements in the tuple. */
  constexpr NDimType size() const noexcept { return size_; }

  /** Return a raw pointer to the tuple. */
  T* data() noexcept { return data_.data(); }

  const T* data() const noexcept { return data_.data(); }

  /** Return a constant raw pointer to the tuple. */
  const T* const_data() const noexcept { return data_.data(); }

  /** Return the value of the tuple at the i'th index. */
  constexpr T operator[](NDimType i) const H2_NOEXCEPT_EXCEPT_DEBUG {
    H2_ASSERT_DEBUG(i < size_, "Tuple index too large");
    return data_[i];
  }

  /**
   * Compare two tuples for equaltiy.
   *
   * Tuples are equal if they have the same size and all valid elements
   * are equal. Note they do not need to have the same maximum size or
   * type; the types just need to be comparable.
   */
  template <typename U, NDimType M>
  constexpr bool operator==(const FixedSizeTuple<U, M>& other) const noexcept {
    if (size_ != other.size_) {
      return false;
    }
    for (NDimType i = 0; i < size_; ++i) {
      if (data_[i] != other.data_[i]) {
        return false;
      }
    }
    return true;
  }
};

/**
 * The shape of a tensor, a tuple with some number of integral values.
 */
using ShapeTuple = FixedSizeTuple<DimType, MAX_TENSOR_DIMS>;

/**
 * Tuple of dimension types.
 */
using DimensionTypeTuple = FixedSizeTuple<DimensionType, MAX_TENSOR_DIMS>;

using DTTuple = DimensionTypeTuple;  // Alias to save you some typing.

/** Product reduction for a tuple. */
template <typename AccT, typename T, NDimType N>
constexpr AccT product(const FixedSizeTuple<T, N>& tuple) noexcept {
  AccT r = AccT{1};
  for (NDimType i = 0; i < tuple.size(); ++i) {
    r *= AccT{tuple[i]};
  }
  return r;
}

/**
 * Represents a contiguous range [start, end).
 */
struct DimensionRange {
  DimType start;  /**< Start of a range. */
  DimType end;  /**< End of a range. */

  constexpr DimensionRange() : start(0), end(0) {}
  constexpr DimensionRange(DimType i) : start(i), end(i+1) {}
  constexpr DimensionRange(DimType start_, DimType end_) : start(start_), end(end_) {}

  constexpr bool operator==(const DimensionRange& other) {
    return start == other.start && end == other.end;
  }
};

/**
 * Tuple of dimension ranges.
 */
using CoordTuple = FixedSizeTuple<DimensionRange, MAX_TENSOR_DIMS>;

/** Special DimensionRange that represents a entire range. */
static constexpr DimensionRange ALL(0, -1);

/**
 * Tuple of exact coordinates.
 */
using SingleCoordTuple = FixedSizeTuple<DimType, MAX_TENSOR_DIMS>;

/**
 * Base class for n-dimensional tensors.
 */
template <typename T>
class BaseTensor {
public:

  using value_type = T;

  /** Construct a tensor with the given shape and dimension types. */
  BaseTensor(ShapeTuple shape_, DimensionTypeTuple dim_types_) :
    tensor_shape(shape_), tensor_dim_types(dim_types_), tensor_is_view(false)
  {}

  /** Construct an empty tensor. */
  BaseTensor() : BaseTensor(ShapeTuple(), DimensionTypeTuple()) {}

  /** Return the shape of the tensor. */
  ShapeTuple shape() const noexcept {
    return tensor_shape;
  }

  /** Return the size of a particular dimension. */
  DimType shape(NDimType i) const noexcept {
    return tensor_shape[i];
  }

  /** Return the types of each dimension of the tensor. */
  DimensionTypeTuple dim_types() const noexcept {
    return tensor_dim_types;
  }

  /** Return the type of a particular dimension. */
  DimensionType dim_type(NDimType i) const noexcept {
    return tensor_dim_types[i];
  }

  /** Return the number of dimensions (i.e., the rank) of the tensor. */
  NDimType ndim() const noexcept {
    return tensor_shape.size();
  }

  /** Return the number of elements in the tensor. */
  DataIndexType numel() const noexcept {
    return product<DataIndexType>(tensor_shape);
  }

  /** Return true if the tensor is empty (all dimensions size 0). */
  bool is_empty() const noexcept {
    return numel() == 0;
  }

  /** Return true if the tensor's underlying memory is contiguous. */
  virtual bool is_contiguous() const noexcept = 0;

  /** Return true if this tensor is a view (i.e., does not own its storage). */
  bool is_view() const noexcept {
    return tensor_is_view;
  }

  /** Return the type of device this tensor is on. */
  virtual Device get_device() const noexcept = 0;

  /** Clear the tensor and reset it to empty. */
  virtual void empty() = 0;

  /** Resize the tensor to a new shape, keeping dimension types the same. */
  virtual void resize(ShapeTuple new_shape) = 0;

  /** Resize the tensor to a new shape, also changing dimension types. */
  virtual void resize(ShapeTuple new_shape, DimensionTypeTuple new_dim_types) = 0;

  /** Return a raw pointer to the underlying storage. */
  virtual T* buffer() = 0;

  /** Return a raw constant pointer to the underlying storage. */
  virtual const T* const_buffer() const = 0;

  /**
   * Make this tensor contiguous, if it is not already.
   *
   * @warning If the tensor is not contiguous, this may invalidate all
   * views.
   */
  virtual void contiguous() = 0;

  /** Return a view of this tensor. */
  virtual BaseTensor<T>* view() = 0;

  /** Return a view of a subtensor of this tensor. */
  virtual BaseTensor<T>* view(CoordTuple coords) = 0;

  /** Convenience wrapper for view(coords). */
  virtual BaseTensor<T>* operator()(CoordTuple coords) {
    return view(coords);
  }

  /**
   * Return the value at a particular coordinate.
   */
  virtual T get(SingleCoordTuple coords) = 0;

protected:
  ShapeTuple tensor_shape;  /**< Shape of the tensor. */
  DimensionTypeTuple tensor_dim_types;  /**< Type of each dimension. */
  bool tensor_is_view;  /**< True if this tensor is a view. */
};

}  // namespace h2
