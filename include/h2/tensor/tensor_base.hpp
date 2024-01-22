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

namespace h2
{

/** For controlling `BaseTensor::ensure`. */
enum class TensorEnsure
{
  NoRecovery,
  AttemptRecovery
};

/**
 * Base class for n-dimensional tensors.
 *
 * Tensors may be allocated lazily: This means that all their metadata
 * is created, but their underlying memory is not allocated until it is
 * needed. `ensure` may be used to guarantee that a tensor has memory
 * backing it. `release` may be used to indicate that the tensor may
 * free its underlying memory (however, memory will not be actually
 * freed until all references to it, e.g., views, have been destroyed
 * or have released their memory).
 *
 * A view of a lazy tensor functions as usual. `ensure` may be called
 * on either the original tensor or the view to allocate memory, even
 * if the view is const.
 *
 * If a tensor `release`s its memory, then calls `ensure` while its
 * original memory remains (e.g., due to a still-extant view), `ensure`
 * can optionally "recover" the original memory. This ensures that any
 * views of the original tensor remain in sync. (This is the default.)
 *
 * Certain operations may implicitly call `ensure` (e.g., `data`).
 * This only happens when the tensor is not const.
 */
template <typename T>
class BaseTensor {
public:

  using value_type = T;

  /**
   * Construct a tensor with the given shape and dimension types.
   */
  BaseTensor(ShapeTuple shape_, DimensionTypeTuple dim_types_) :
    tensor_shape(shape_),
    tensor_dim_types(dim_types_),
    tensor_view_type(ViewType::None)
  {
    H2_ASSERT_DEBUG(tensor_shape.size() == tensor_dim_types.size(),
                    "Tensor shape and dimension types must be the same size");
  }

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
  virtual typename StrideTuple::type stride(typename StrideTuple::size_type i) const H2_NOEXCEPT = 0;

  /** Return the number of dimensions (i.e., the rank) of the tensor. */
  typename ShapeTuple::size_type ndim() const H2_NOEXCEPT {
    return tensor_shape.size();
  }

  /** Return the number of elements in the tensor. */
  DataIndexType numel() const H2_NOEXCEPT {
    if (tensor_shape.empty()) {
      return 0;
    }
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

  /** Return true if this tensor is a constant view. */
  bool is_const_view() const H2_NOEXCEPT {
    return tensor_view_type == ViewType::Const;
  }

  /** Return the type of view this tensor is (may be ViewType::None). */
  ViewType get_view_type() const H2_NOEXCEPT { return tensor_view_type; }

  /** Return the type of device this tensor is on. */
  virtual Device get_device() const H2_NOEXCEPT = 0;

  /**
   * Clear the tensor and reset it to empty.
   *
   * If this is a view, this is equivalent to `unview`.
   */
  virtual void empty() = 0;

  /**
   * Resize the tensor to a new shape, keeping dimension types the same.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(ShapeTuple new_shape) = 0;

  /**
   * Resize the tensor to a new shape, also changing dimension types.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(ShapeTuple new_shape, DimensionTypeTuple new_dim_types) = 0;

  /**
   * Return a raw pointer to the underlying storage.
   *
   * @note Remember to account for the strides when accessing this.
   */
  virtual T* data() = 0;

  /** Return a raw constant pointer to the underlying storage. */
  virtual const T* data() const = 0;

  /** Return a raw constant pointer to the underlying storage. */
  virtual const T* const_data() const = 0;

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * Optionally attempt to reuse existing memory from still-extant
   * views of this tensor.
   */
  virtual void ensure(
    TensorEnsure attempt_recover = TensorEnsure::AttemptRecovery) = 0;

  /**
   * Release memory associated with this tensor.
   *
   * Note that if there are views, memory may not be deallocated
   * immediately.
   */
  virtual void release() = 0;

  /**
   * Return a contiguous version of this tensor.
   *
   * If the tensor is contiguous, a view of the original tensor is
   * returned. Otherwise, a new tensor is allocated.
   *
   * If this tensor is a view, the returned tensor will be distinct
   * from the viewed tensor. Any views of this tensor will still be
   * viewing the original tensor, not the contiguous tensor.
   */
  virtual BaseTensor<T>* contiguous() = 0;

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
   * underlying memory (e.g., `resize`) are not permitted on views and
   * will throw an exception. Other operations have special semantics
   * when the tensor is a view (e.g., `contiguous`, `empty`).
   */
  virtual BaseTensor<T>* view() = 0;

  /** Return a constant view of this tensor. */
  virtual BaseTensor<T>* view() const = 0;

  /**
   * Return a view of a subtensor of this tensor.
   *
   * Note that (inherent in the definition of `DimensionRange`), views
   * must be of contiguous subsets of the tensor (i.e., no strides).
   */
  virtual BaseTensor<T>* view(CoordTuple coords) = 0;

  /**
   * Return a constant view of a subtensor of this tensor.
   */
  virtual BaseTensor<T>* view(CoordTuple coords) const = 0;

  /**
   * If this tensor is a view, stop viewing.
   *
   * The tensor will have empty dimensions after this.
   *
   * It is an error to call this if the tensor is not a view.
   */
  virtual void unview() = 0;

  // Note: The operator() is abstract rather than defaulting to
  // view(coords) because we need to covariant return type.

  /** Convenience wrapper for view(coords). */
  virtual BaseTensor<T>* operator()(CoordTuple coords) = 0;

  /** Return a constant view of this tensor. */
  virtual BaseTensor<T>* const_view() const = 0;

  /** Return a constant view of a subtensor of this tensor. */
  virtual BaseTensor<T>* const_view(CoordTuple coords) const = 0;

  /** Convenience wrapper for const_view(coords). */
  virtual BaseTensor<T>* operator()(CoordTuple coords) const = 0;

  /** Return a pointer to the tensor at a particular coordinate. */
  virtual T* get(SingleCoordTuple coords) = 0;

  /**
   * Return a constant pointer to the tensor at a particular coordinate.
   */
  virtual const T* get(SingleCoordTuple coords) const = 0;

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

}  // namespace h2
