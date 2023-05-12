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

#include "h2/tensor/tensor_types.hpp"
#include "tensor_types.hpp"

namespace h2
{

/**
 * Base class for n-dimensional tensors.
 */
template <typename T>
class BaseTensor {
public:

  using value_type = T;

  /** Construct a tensor with the given shape and dimension types. */
  BaseTensor(ShapeTuple shape_, DimensionTypeTuple dim_types_) :
    tensor_shape(shape_),
    tensor_dim_types(dim_types_),
    tensor_view_type(ViewType::None)
  {}

  /** Construct an empty tensor. */
  BaseTensor() : BaseTensor(ShapeTuple(), DimensionTypeTuple()) {}

  /** Return the shape of the tensor. */
  ShapeTuple shape() const H2_NOEXCEPT {
    return tensor_shape;
  }

  /** Return the size of a particular dimension. */
  typename ShapeTuple::type shape(typename ShapeTuple::size_type i) const H2_NOEXCEPT {
    return tensor_shape[i];
  }

  /** Return the types of each dimension of the tensor. */
  DimensionTypeTuple dim_types() const H2_NOEXCEPT {
    return tensor_dim_types;
  }

  /** Return the type of a particular dimension. */
  typename DimensionTypeTuple::type dim_type(typename DimensionTypeTuple::size_type i) const H2_NOEXCEPT {
    return tensor_dim_types[i];
  }

  /** Return the strides of the underlying memory for the tensor. */
  virtual StrideTuple strides() const H2_NOEXCEPT = 0;

  /** Return the stride of a particular dimension. */
  virtual typename StrideTuple::type strides(typename StrideTuple::size_type i) const H2_NOEXCEPT = 0;

  /** Return the number of dimensions (i.e., the rank) of the tensor. */
  typename ShapeTuple::size_type ndim() const H2_NOEXCEPT {
    return tensor_shape.size();
  }

  /** Return the number of elements in the tensor. */
  DataIndexType numel() const H2_NOEXCEPT {
    return product<DataIndexType>(tensor_shape);
  }

  /** Return true if the tensor is empty (all dimensions size 0). */
  bool is_empty() const H2_NOEXCEPT {
    return numel() == 0;
  }

  /** Return true if the tensor's underlying memory is contiguous. */
  virtual bool is_contiguous() const H2_NOEXCEPT = 0;

  /** Return true if this tensor is a view (i.e., does not own its storage). */
  bool is_view() const H2_NOEXCEPT {
    return tensor_view_type != ViewType::None;
  }

  /** Return the type of device this tensor is on. */
  virtual Device get_device() const H2_NOEXCEPT = 0;

  /** Clear the tensor and reset it to empty. */
  virtual void empty() = 0;

  /** Resize the tensor to a new shape, keeping dimension types the same. */
  virtual void resize(ShapeTuple new_shape) = 0;

  /** Resize the tensor to a new shape, also changing dimension types. */
  virtual void resize(ShapeTuple new_shape, DimensionTypeTuple new_dim_types) = 0;

  /** Return a raw pointer to the underlying storage. */
  virtual T* data() = 0;

  /** Return a raw constant pointer to the underlying storage. */
  virtual const T* const_data() const = 0;

  /**
   * Make this tensor contiguous, if it is not already.
   *
   * This may reallocate the tensor's memory.
   *
   * @warning If the tensor is not contiguous, this may invalidate all
   * views.
   */
  //virtual void contiguous() = 0;

  /**
   * Return a view of this tensor.
   *
   * A view will share the same underlying memory as the original tensor,
   * and will therefore reflect any changes made to the data; likewise,
   * changes made through the view will be reflected in the original
   * tensor. Additionally, the memory will remain valid so long as the
   * view exists, even if the original tensor no longer exists.
   *
   * However, changes to metadata (e.g., shape, stride, etc.) do not
   * propagate to views. It is up to the caller to ensure views remain
   * consistent. Certain operations that would require changes to the
   * underlying memory (`resize`, `contiguous`, `empty`, etc.) are not
   * permitted on views and will throw an exception.
   */
  virtual BaseTensor<T>* view() = 0;

  /**
   * Return a view of a subtensor of this tensor.
   *
   * Note that (inherent in the definition of `DimensionRange`), views
   * must be of contiguous subsets of the tensor (i.e., no strides).
   */
  virtual BaseTensor<T>* view(CoordTuple coords) = 0;

  /** Convenience wrapper for view(coords). */
  virtual BaseTensor<T>* operator()(CoordTuple coords) {
    return view(coords);
  }

  /** Return a constant view of this tensor. */
  virtual const BaseTensor<T>* const_view() const = 0;

  /** Return a constant view of a subtensor of this tensor. */
  virtual const BaseTensor<T>* const_view(CoordTuple coords) const = 0;

  /** Convenience wrapper for const_view(coords). */
  virtual const BaseTensor<T>* operator()(CoordTuple coords) const {
    return const_view(coords);
  }

  /**
   * Return the value at a particular coordinate.
   */
  virtual T get(SingleCoordTuple coords) const = 0;

protected:
  ShapeTuple tensor_shape;  /**< Shape of the tensor. */
  DimensionTypeTuple tensor_dim_types;  /**< Type of each dimension. */
  ViewType tensor_view_type;  /**< What type of view (if any) this tensor is. */

  /** Construct a tensor with the given view type, shape, and dimension types. */
  BaseTensor(ViewType view_type_,
             ShapeTuple shape_,
             DimensionTypeTuple dim_types_) :
    tensor_shape(shape_),
    tensor_dim_types(dim_types_),
    tensor_view_type(view_type_) {}
};

/** Generic Tensor class for arbitrary types. */
template <typename T, Device Dev>
class Tensor;

}  // namespace h2

#include "h2/tensor/tensor_cpu.hpp"
