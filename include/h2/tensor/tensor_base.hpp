////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * This defines the public API for local (non-distributed) tensors.
 */

#include "h2/core/types.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/Describable.hpp"
#include "h2/utils/typename.hpp"

namespace h2
{

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
class BaseTensor : public Describable
{
public:
  /**
   * Construct a tensor with the given shape and dimension types.
   */
  BaseTensor(const ShapeTuple& shape_, const DimensionTypeTuple& dim_types_)
    : tensor_shape(shape_),
      tensor_dim_types(dim_types_),
      tensor_view_type(ViewType::None)
  {
    H2_ASSERT_DEBUG(tensor_shape.size() == tensor_dim_types.size(),
                    "Tensor shape (",
                    tensor_shape,
                    " ) and dimension types (",
                    tensor_dim_types,
                    ") must be the same size");
    H2_ASSERT_DEBUG(shape_.is_empty() || product<DataIndexType>(shape_) > 0,
                    "Zero-length dimensions are not permitted, got ",
                    shape_);
  }

  /** Construct an empty tensor. */
  BaseTensor() : BaseTensor(ShapeTuple(), DimensionTypeTuple()) {}

  virtual ~BaseTensor() = default;

  /** Return information on the type the tensor stores. */
  virtual TypeInfo get_type_info() const H2_NOEXCEPT = 0;

  /** Return the shape of the tensor. */
  ShapeTuple shape() const H2_NOEXCEPT { return tensor_shape; }

  /** Return the size of a particular dimension. */
  typename ShapeTuple::type
  shape(typename ShapeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_shape[i];
  }

  /** Return the types of each dimension of the tensor. */
  DimensionTypeTuple dim_types() const H2_NOEXCEPT { return tensor_dim_types; }

  /** Return the type of a particular dimension. */
  typename DimensionTypeTuple::type
  dim_type(typename DimensionTypeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_dim_types[i];
  }

  /** Return the strides of the underlying memory for the tensor. */
  virtual StrideTuple strides() const H2_NOEXCEPT = 0;

  /** Return the stride of a particular dimension. */
  virtual typename StrideTuple::type
  stride(typename StrideTuple::size_type i) const H2_NOEXCEPT = 0;

  /** Return the number of dimensions (i.e., the rank) of the tensor. */
  typename ShapeTuple::size_type ndim() const H2_NOEXCEPT
  {
    return tensor_shape.size();
  }

  /** Return the number of elements in the tensor. */
  DataIndexType numel() const H2_NOEXCEPT
  {
    if (tensor_shape.is_empty())
    {
      return 0;
    }
    return product<DataIndexType>(tensor_shape);
  }

  /** Return true if the tensor is empty (all dimensions size 0). */
  bool is_empty() const H2_NOEXCEPT { return numel() == 0; }

  /** Return true if the tensor's underlying memory is contiguous. */
  virtual bool is_contiguous() const H2_NOEXCEPT = 0;

  /** Return true if this tensor is a view (i.e., does not own its storage). */
  bool is_view() const H2_NOEXCEPT
  {
    return tensor_view_type != ViewType::None;
  }

  /** Return true if this tensor is a constant view. */
  bool is_const_view() const H2_NOEXCEPT
  {
    return tensor_view_type == ViewType::Const;
  }

  /** Return the type of view this tensor is (may be ViewType::None). */
  ViewType get_view_type() const H2_NOEXCEPT { return tensor_view_type; }

  /** Return the type of device this tensor is on. */
  virtual Device get_device() const H2_NOEXCEPT = 0;

  /** Get the compute stream associated with this tensor. */
  virtual ComputeStream get_stream() const H2_NOEXCEPT = 0;

  /** Set the compute stream associated with this tensor. */
  virtual void set_stream(const ComputeStream& stream) = 0;

  /**
   * Return a raw, generic pointer (void*) to the underlying storage.
   *
   * This is meant for working with storage (as opposed to compute)
   * types.
   */
  virtual void* storage_data() = 0;

  /**
   * Return a raw, generic constant pointer (const void*) to the
   * underlying storage.
   */
  virtual const void* storage_data() const = 0;

  /**
   * Return a raw, generic constant pointer (const void*) to the
   * underlying storage.
   */
  virtual const void* const_storage_data() const = 0;

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This attempts to reuse existing memory from still-extant views of
   * this tensor.
   */
  virtual void ensure() = 0;

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This does not attempt to reuse existing memory from still-extant
   * views of this tensor.
   */
  virtual void ensure(tensor_no_recovery_t) = 0;

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This attempts to reuse existing memory from still-extant views of
   * this tensor.
   */
  virtual void ensure(tensor_attempt_recovery_t) = 0;

  /**
   * Release memory associated with this tensor.
   *
   * Note that if there are views, memory may not be deallocated
   * immediately.
   */
  virtual void release() = 0;

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
  virtual void resize(const ShapeTuple& new_shape) = 0;

  /**
   * Resize the tensor to a new shape, also changing dimension types.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(const ShapeTuple& new_shape,
                      const DimensionTypeTuple& new_dim_types) = 0;

  /**
   * Resize the tensor to a new shape, also changing dimension types
   * and specifying new strides.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(const ShapeTuple& new_shape,
                      const DimensionTypeTuple& new_dim_types,
                      const StrideTuple& new_strides) = 0;

protected:
  ShapeTuple tensor_shape;             /**< Shape of the tensor. */
  DimensionTypeTuple tensor_dim_types; /**< Type of each dimension. */
  ViewType tensor_view_type; /**< What type of view (if any) this tensor is. */

  /** Construct a tensor with the given view type, shape, and dimension types.
   */
  BaseTensor(ViewType view_type_,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_)
    : tensor_shape(shape_),
      tensor_dim_types(dim_types_),
      tensor_view_type(view_type_)
  {
    H2_ASSERT_DEBUG(tensor_shape.size() == tensor_dim_types.size(),
                    "Tensor shape ",
                    tensor_shape,
                    " and dimension types ",
                    tensor_dim_types,
                    " must be the same size");
  }
};

}  // namespace h2
