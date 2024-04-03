////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * This defines the public API for distributed tensors.
 */

#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/dist_utils.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "h2/tensor/tensor_base.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/Describable.hpp"
#include "h2/utils/typename.hpp"


namespace h2
{

/**
 * Base class for n-dimensional distributed tensors.
 *
 * A distributed tensor is partitioned over a processor grid according
 * to a provided distribution. The processor grid must have the same
 * number of dimensions as the tensor. Each dimension of the tensor may
 * be distributed with a different distribution.
 *
 * The partitioning is determined by partitioning the indices of each
 * dimension independently, according to that dimension's assigned
 * distribution. If any dimension is assigned zero indices for a rank,
 * that rank will not be assigned any data (and will have an empty
 * local tensor).
 *
 * Distributed tensors generally follow the same semantics as
 * `BaseTensor`s and `Tensor`s.
 *
 * A key exception is views of distributed tensors, which may not
 * eliminate any dimensions.
 */
template <typename T>
class BaseDistTensor : public Describable
{
public:

  using value_type = T;

  /**
   * Construct a tensor with the given shape and dimension types,
   * distributed over the given processor grid.
   */
  BaseDistTensor(const ShapeTuple& shape_,
                 const DimensionTypeTuple& dim_types_,
                 ProcessorGrid grid_,
                 const DistributionTypeTuple& dist_types_)
      : tensor_shape(shape_),
        tensor_dim_types(dim_types_),
        tensor_grid(grid_),
        tensor_dist_types(dist_types_),
        tensor_view_type(ViewType::None)
  {
    H2_ASSERT_DEBUG(tensor_shape.size() == tensor_dim_types.size(),
                    "Tensor shape and dimension types must be the same size");
    H2_ASSERT_DEBUG((tensor_shape.size() == tensor_grid.ndim())
                        || tensor_shape.is_empty(),
                    "Tensor and processor grid must be the same rank");
    H2_ASSERT_DEBUG(
        tensor_shape.size() == tensor_dist_types.size(),
        "Tensor distribution types and processor grid must be the same rank");
    tensor_local_shape =
        internal::get_local_shape(tensor_shape, tensor_grid, tensor_dist_types);
  }

  /** Construct an empty tensor on a null grid. */
  BaseDistTensor()
      : BaseDistTensor(ShapeTuple{},
                       DimensionTypeTuple{},
                       ProcessorGrid{},
                       DistributionTypeTuple{})
  {}

  virtual ~BaseDistTensor() = default;

  /** Return the shape of the tensor. */
  ShapeTuple shape() const H2_NOEXCEPT
  {
    return tensor_shape;
  }

  /** Return the size of a particular dimension. */
  typename ShapeTuple::type
  shape(typename ShapeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_shape[i];
  }

  /** Return the local shape of the tensor. */
  ShapeTuple local_shape() const H2_NOEXCEPT
  {
    return tensor_local_shape;
  }

  /** Return the local size of a particular dimension. */
  typename ShapeTuple::type
  local_shape(typename ShapeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_local_shape[i];
  }

  /** Return the types of each dimension of the tensor. */
  DimensionTypeTuple dim_types() const H2_NOEXCEPT
  {
    return tensor_dim_types;
  }

  /** Return the type of a particular dimension. */
  typename DimensionTypeTuple::type
  dim_type(typename DimensionTypeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_dim_types[i];
  }

  /** Return the tensor's processor grid. */
  ProcessorGrid proc_grid() const H2_NOEXCEPT { return tensor_grid; }

  /** Return the distribution types of the tensor. */
  DistributionTypeTuple distribution() const H2_NOEXCEPT
  {
    return tensor_dist_types;
  }

  /** Return the distribution for a particular dimension. */
  Distribution
  distribution(typename DistributionTypeTuple::size_type i) const H2_NOEXCEPT
  {
    return tensor_dist_types[i];
  }

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

  /** Return the number of elements in the local tensor. */
  DataIndexType local_numel() const H2_NOEXCEPT
  {
    if (tensor_local_shape.is_empty())
    {
      return 0;
    }
    return product<DataIndexType>(tensor_local_shape);
  }

  /**
   * Return true if the tensor is empty (all dimensions size 0).
   *
   * @note This refers to the global tensor; the local storage may
   * still be empty, depending on the distribution.
   */
  bool is_empty() const H2_NOEXCEPT { return numel() == 0; }

  /**
   * Return true if the local tensor is empty (all dimensions size 0).
   */
  bool is_local_empty() const H2_NOEXCEPT { return local_numel() == 0; }

  /** Output a short description of the tensor. */
  void short_describe(std::ostream& os) const override
  {
    os << "DistTensor<" << TypeName<T>() << ", " << get_device() << ">(";
    tensor_grid.short_describe(os);
    os << ": ";
    if (is_view())
    {
      os << get_view_type() << " of ";
    }
    for (ShapeTuple::size_type i = 0; i < ndim(); ++i)
    {
      os << distribution(i) << ":" << dim_type(i) << ":" << shape(i);
      if (i < ndim() - 1)
      {
        os << " x ";
      }
    }
    os << ")";
  }

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

  /**
   * Clear the tensor and reset it to empty.
   *
   * If this is a view, this is equivalent to `unview`.
   */
  virtual void empty() = 0;

  /**
   * Resize the tensor to a new shape.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(const ShapeTuple& new_shape) = 0;

  /**
   * Resize the tensor to a new shape and change its dimension types.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(const ShapeTuple& new_shape,
                      const DimensionTypeTuple& new_dim_types) = 0;

  /**
   * Resize the tensor to a new shape and change its dimension types
   * and distribution.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  virtual void resize(const ShapeTuple& new_shape,
                      const DimensionTypeTuple& new_dim_types,
                      const DistributionTypeTuple& new_dist_types) = 0;

  /**
   * Return a raw pointer to the underlying local storage.
   *
   * @note Remember to account for the strides when accessing this.
   * @note Just because a tensor is globally non-empty does not mean it
   * has local data.
   */
  virtual T* data() = 0;

  /** Return a raw constant pointer to the underlying local storage. */
  virtual const T* data() const = 0;

  /** Return a raw constant pointer to the underlying local storage. */
  virtual const T* const_data() const = 0;

  /** Return the underlying local tensor. */
  virtual BaseTensor<T>& local_tensor() = 0;

  /** Return a constant reference to the underlying local tensor. */
  virtual const BaseTensor<T>& local_tensor() const = 0;

  /** Return a constant reference to the underlying local tensor. */
  virtual const BaseTensor<T>& const_local_tensor() const = 0;

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
  virtual BaseDistTensor<T>* view() = 0;

  /** Return a constant view of this tensor. */
  virtual BaseDistTensor<T>* view() const = 0;

  /**
   * Return a view of a subtensor of this tensor.
   *
   * Note that (inherent in the definition of `IndexRange`), views
   * must be of contiguous subsets of the tensor (i.e., no strides).
   *
   * The `coords` given may omit dimensions on the right. In this case,
   * they are assumed to have their full range. However, if `coords` is
   * fully empty, the view will be empty.
   *
   * No entries of `coords` may be scalar: Dimensions cannot be
   * eliminated from a distributed tensor, unless the entire view is
   * empty.
   */
  virtual BaseDistTensor<T>* view(const IndexRangeTuple& coords) = 0;

  /**
   * Return a constant view of a subtensor of this tensor.
   */
  virtual BaseDistTensor<T>* view(const IndexRangeTuple& coords) const = 0;

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
  virtual BaseDistTensor<T>* operator()(const IndexRangeTuple& coords) = 0;

  /** Return a constant view of this tensor. */
  virtual BaseDistTensor<T>* const_view() const = 0;

  /** Return a constant view of a subtensor of this tensor. */
  virtual BaseDistTensor<T>*
  const_view(const IndexRangeTuple& coords) const = 0;

  /** Convenience wrapper for const_view(coords). */
  virtual BaseDistTensor<T>*
  operator()(const IndexRangeTuple& coords) const = 0;

protected:
  ShapeTuple tensor_shape;  /**< Global shape of the tensor. */
  DimensionTypeTuple tensor_dim_types;  /**< Type of each dimension. */
  ProcessorGrid tensor_grid;  /**< Grid the tensor is distributed over. */
  /** How each dimension of the tensor is distributed. */
  DistributionTypeTuple tensor_dist_types;
  ViewType tensor_view_type;  /**< What type of view (if any) this tensor is. */

  // Implementation note:
  // This somewhat duplicates size information in the concrete
  // `DistTensor` implementation (which has an internal `Tensor` that
  // of course also stores its shape). Keeping this here allows us to
  // compute local size information while remaining abstract.

  ShapeTuple tensor_local_shape;  /**< Local shape of the tensor. */

  /** Construct a tensor with the given view type. */
  BaseDistTensor(ViewType view_type_,
                 const ShapeTuple& shape_,
                 const DimensionTypeTuple& dim_types_,
                 ProcessorGrid grid_,
                 const DistributionTypeTuple& dist_types_,
                 const ShapeTuple& local_shape_)
      : tensor_shape(shape_),
        tensor_dim_types(dim_types_),
        tensor_grid(grid_),
        tensor_dist_types(dist_types_),
        tensor_view_type(view_type_),
        tensor_local_shape(local_shape_) {}
};

}  // namespace h2
