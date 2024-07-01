////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Distributed tensors that live on a device.
 */

#include <memory>
#include <optional>

#include "h2/tensor/dist_tensor_base.hpp"
#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "h2/tensor/tensor.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/passkey.hpp"

namespace h2
{

/** Distributed tensor class for arbitrary types and devices. */
template <typename T>
class DistTensor : public BaseDistTensor
{
public:

  using value_type = T;
  using local_tensor_type = Tensor<T>;

  DistTensor(Device device,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             TensorAllocationStrategy alloc_type = StrictAlloc,
             const std::optional<ComputeStream> stream = std::nullopt)
      : BaseDistTensor(shape_, dim_types_, grid_, dist_types_),
        tensor_local(device,
                     this->tensor_local_shape,
                     init_n(dim_types_, this->tensor_local_shape.size()),
                     alloc_type,
                     stream.value_or(ComputeStream{device}))
  {}

  DistTensor(Device device,
             ProcessorGrid grid_,
             TensorAllocationStrategy alloc_type = StrictAlloc,
             const std::optional<ComputeStream> stream = std::nullopt)
      : DistTensor(device,
                   ShapeTuple(),
                   DimensionTypeTuple(),
                   grid_,
                   DistributionTypeTuple(),
                   alloc_type,
                   stream)
  {}

  DistTensor(Device device,
             T* buffer,
             const ShapeTuple& global_shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             const ShapeTuple& local_shape_,
             const StrideTuple& local_strides_,
             const ComputeStream& stream)
      : BaseDistTensor(ViewType::Mutable,
                       global_shape_,
                       dim_types_,
                       grid_,
                       dist_types_,
                       local_shape_),
        tensor_local(
            device, buffer, local_shape_, dim_types_, local_strides_, stream)
  {}

  DistTensor(Device device,
             const T* buffer,
             const ShapeTuple& global_shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             const ShapeTuple& local_shape_,
             const StrideTuple& local_strides_,
             const ComputeStream& stream)
      : BaseDistTensor(ViewType::Const,
                          global_shape_,
                          dim_types_,
                          grid_,
                          dist_types_,
                          local_shape_),
        tensor_local(
            device, buffer, local_shape_, dim_types_, local_strides_, stream)
  {}

  /** Internal constructor for views. */
  DistTensor(ViewType view_type_,
             const Tensor<T>& orig_tensor_local_,
             const ShapeTuple& local_shape_,
             const IndexRangeTuple& local_coords_,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             Passkey<DistTensor<T>>)
      : BaseDistTensor(
          view_type_, shape_, dim_types_, grid_, dist_types_, local_shape_),
        tensor_local(view_type_,
                     orig_tensor_local_.tensor_memory,
                     local_shape_,
                     // Local shape may be empty depending on the view
                     // coordinates and distribution (e.g., Single).
                     local_shape_.is_empty() ? DimensionTypeTuple{}
                                             : dim_types_,
                     local_coords_,
                     Passkey<DistTensor<T>>{})
  {}

  /** Internal constructor for cloning. */
  DistTensor(Tensor<T>& local_tensor_clone,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             Passkey<DistTensor<T>>)
      : BaseDistTensor(ViewType::None,
                       shape_,
                       dim_types_,
                       grid_,
                       dist_types_,
                       local_tensor_clone.shape()),
        tensor_local(std::move(local_tensor_clone))
  {}

  virtual ~DistTensor() = default;

  /**
   * Disable copy construction.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  DistTensor(const DistTensor&) = delete;

  /**
   * Disable copy assignment.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  DistTensor& operator=(const DistTensor&) = delete;

  /** Move construction */
  DistTensor(DistTensor&&) = default;

  /** Move assignment */
  DistTensor& operator=(DistTensor&&) = default;

  /**
   * Return an exact copy of this tensor.
   *
   *
   */
  std::unique_ptr<DistTensor<T>> clone() const
  {
    auto local_tensor_clone = tensor_local.clone();
    return std::make_unique<DistTensor<T>>(*local_tensor_clone,
                                           shape(),
                                           dim_types(),
                                           proc_grid(),
                                           distribution(),
                                           Passkey<DistTensor<T>>{});
  }

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

  Device get_device() const H2_NOEXCEPT override
  {
    return tensor_local.get_device();
  }

  /**
   * Clear the tensor and reset it to empty.
   *
   * If this is a view, this is equivalent to `unview`.
   */
  void empty()
  {
    this->tensor_local.empty();
    this->tensor_shape = ShapeTuple();
    this->tensor_dim_types = DimensionTypeTuple();
    this->tensor_dist_types = DistributionTypeTuple();
    this->tensor_local_shape = ShapeTuple();
    if (this->is_view())
    {
      this->tensor_view_type = ViewType::None;
    }
  }

  /**
   * Resize the tensor to a new shape.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape)
  {
    resize(new_shape, this->tensor_dim_types, this->tensor_dist_types);
  }

  /**
   * Resize the tensor to a new shape and change its dimension types.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types)
  {
    resize(new_shape, new_dim_types, this->tensor_dist_types);
  }

  /**
   * Resize the tensor to a new shape and change its dimension types
   * and distribution.
   *
   * The number of dimensions must be the same as the existing tensor,
   * or the existing tensor must be empty.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types,
              const DistributionTypeTuple& new_dist_types)
  {
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_shape.size() == this->tensor_grid.ndim(),
                     "Cannot change the number of dimensions when resizing a "
                     "distributed tensor");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "Shape (",
                     new_shape,
                     ") and dimension types (",
                     new_dim_types,
                     ") must be the same size");
    H2_ASSERT_ALWAYS(new_dist_types.size() == new_shape.size(),
                     "Shape (",
                     new_shape,
                     ") and distribution types (",
                     new_dist_types,
                     ") must be the same size");
    this->tensor_shape = new_shape;
    this->tensor_local_shape = internal::get_local_shape(
        new_shape, this->tensor_grid, new_dist_types);
    this->tensor_dim_types = new_dim_types;
    this->tensor_dist_types = new_dist_types;
    tensor_local.resize(this->tensor_local_shape,
                        init_n(new_dim_types, this->tensor_local_shape.size()));
  }

  /**
   * Return a raw pointer to the underlying local storage.
   *
   * @note Remember to account for the strides when accessing this.
   * @note Just because a tensor is globally non-empty does not mean it
   * has local data.
   */
  T* data()
  {
    return tensor_local.data();
  }

  /** Return a raw constant pointer to the underlying local storage. */
  const T* data() const
  {
    return tensor_local.data();
  }

  /** Return a raw constant pointer to the underlying local storage. */
  const T* const_data() const
  {
    return tensor_local.const_data();
  }

  /** Return the underlying local tensor. */
  Tensor<T>& local_tensor()
  {
    return tensor_local;
  }

  /** Return a constant reference to the underlying local tensor. */
  const Tensor<T>& local_tensor() const
  {
    return tensor_local;
  }

  /** Return a constant reference to the underlying local tensor. */
  const Tensor<T>& const_local_tensor() const
  {
    return tensor_local;
  }

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This attempts to reuse existing memory from still-extant views of
   * this tensor.
   */
  void ensure()
  {
    ensure(TensorAttemptRecovery);
  }

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This does not attempt to reuse existing memory from still-extant
   * views of this tensor.
   */
  void ensure(tensor_no_recovery_t)
  {
    tensor_local.ensure(TensorNoRecovery);
  }

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This attempts to reuse existing memory from still-extant views of
   * this tensor.
   */
  void ensure(tensor_attempt_recovery_t)
  {
    tensor_local.ensure(TensorAttemptRecovery);
  }

  /**
   * Release memory associated with this tensor.
   *
   * Note that if there are views, memory may not be deallocated
   * immediately.
   */
  void release()
  {
    tensor_local.release();
  }

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
  std::unique_ptr<DistTensor<T>> view()
  {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  /** Return a constant view of this tensor. */
  std::unique_ptr<DistTensor<T>> view() const
  {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

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
  std::unique_ptr<DistTensor<T>> view(const IndexRangeTuple& coords)
  {
    return make_view(coords, ViewType::Mutable);
  }

  /**
   * Return a constant view of a subtensor of this tensor.
   */
  std::unique_ptr<DistTensor<T>> view(const IndexRangeTuple& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for view(coords). */
  std::unique_ptr<DistTensor<T>> operator()(const IndexRangeTuple& coords)
  {
    return view(coords);
  }

  /**
   * If this tensor is a view, stop viewing.
   *
   * The tensor will have empty dimensions after this.
   *
   * It is an error to call this if the tensor is not a view.
   */
  void unview()
  {
    H2_ASSERT_DEBUG(this->is_view(), "Must be a view to unview");
    empty();  // Emptying a view is equivalent to unviewing.
  }

  /** Return a constant view of this tensor. */
  std::unique_ptr<DistTensor<T>> const_view() const
  {
    return const_view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  /** Return a constant view of a subtensor of this tensor. */
  std::unique_ptr<DistTensor<T>> const_view(const IndexRangeTuple& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for const_view(coords). */
  std::unique_ptr<DistTensor<T>> operator()(const IndexRangeTuple& coords) const
  {
    return const_view(coords);
  }

  ComputeStream get_stream() const H2_NOEXCEPT
  {
    return tensor_local.get_stream();
  }

  void set_stream(const ComputeStream& stream)
  {
    tensor_local.set_stream(stream);
  }

  bool is_lazy() const H2_NOEXCEPT
  {
    return tensor_local.is_lazy();
  }

private:
  /** Local tensor used for storage. */
  Tensor<T> tensor_local;

  /** Helper for constructing views. */
  std::unique_ptr<DistTensor<T>> make_view(IndexRangeTuple index_range,
                                           ViewType view_type) const
  {
    H2_ASSERT_ALWAYS(is_index_range_contained(index_range, this->tensor_shape),
                     "Cannot construct an out-of-range view, ",
                     index_range,
                     " is not contained in ",
                     this->tensor_shape);
    H2_ASSERT_ALWAYS(
        !any_of(index_range,
                [](const IndexRangeTuple::type& x) { return x.is_scalar(); }),
        "Scalar indices (", index_range, ") are not permitted in global views");

    // We have three cases:
    // 1. The indices are empty, so we have a globally empty view.
    // 2. The indices result in a valid global view, but an empty local
    // tensor.
    // 3. The indices result in a valid global view and we have local
    // data.

    if (is_index_range_empty(index_range))
    {
      // Globally empty view.
      return std::make_unique<DistTensor<T>>(view_type,
                                             tensor_local,
                                             ShapeTuple{},
                                             IndexRangeTuple{},
                                             ShapeTuple{},
                                             DimensionTypeTuple{},
                                             this->tensor_grid,
                                             DistributionTypeTuple{},
                                             Passkey<DistTensor<T>>{});
    }

    // Standardize the indices to have all dimensions, adding any ALL
    // dimensions on the right.
    if (index_range.size() < this->tensor_shape.size())
    {
      for (typename IndexRangeTuple::size_type i = index_range.size();
           i < this->tensor_shape.size();
           ++i)
      {
        index_range.append(ALL);
      }
    }

    // Get the global shape of the view.
    ShapeTuple view_global_shape =
        get_index_range_shape(index_range, this->tensor_shape);
    // Get the global indices of the original tensor that this rank
    // owns (i.e., that are present locally).
    IndexRangeTuple global_indices = internal::get_global_indices(
        this->tensor_shape, this->tensor_grid, this->tensor_dist_types);

    if (!do_index_ranges_intersect(index_range, global_indices))
    {
      // This rank has no local data in the view.
      return std::make_unique<DistTensor<T>>(view_type,
                                             tensor_local,
                                             ShapeTuple{},
                                             IndexRangeTuple{},
                                             view_global_shape,
                                             this->tensor_dim_types,
                                             this->tensor_grid,
                                             this->tensor_dist_types,
                                             Passkey<DistTensor<T>>{});
    }

    // Determine this rank's indices in the view by intersecting
    // its global indices with the global indices defining the view.
    IndexRangeTuple present_global_indices =
        intersect_index_ranges(global_indices, index_range);
    // Convert this to local indices for setting up the local view.
    IndexRangeTuple local_indices =
        internal::global2local_indices(this->tensor_shape,
                                      this->tensor_grid,
                                      this->tensor_dist_types,
                                      present_global_indices);
    ShapeTuple view_local_shape =
        get_index_range_shape(local_indices, this->tensor_local_shape);
    return std::make_unique<DistTensor<T>>(view_type,
                                           tensor_local,
                                           view_local_shape,
                                           local_indices,
                                           view_global_shape,
                                           this->tensor_dim_types,
                                           this->tensor_grid,
                                           this->tensor_dist_types,
                                           Passkey<DistTensor<T>>{});
  }

};

}  // namespace h2
