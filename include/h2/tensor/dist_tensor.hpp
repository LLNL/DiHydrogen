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

#include <optional>

#include "h2/tensor/dist_tensor_base.hpp"
#include "h2/tensor/dist_types.hpp"
#include "h2/tensor/proc_grid.hpp"
#include "h2/tensor/tensor.hpp"
#include "h2/tensor/tensor_types.hpp"

namespace h2
{

/** Distributed tensor class for arbitrary types and devices. */
template <typename T>
class DistTensor : public BaseDistTensor<T>
{
public:

  using value_type = T;
  using local_tensor_type = Tensor<T>;

  DistTensor(Device device,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_,
             TensorAllocation alloc_type = StrictAlloc,
             const std::optional<ComputeStream> stream = std::nullopt)
      : BaseDistTensor<T>(shape_, dim_types_, grid_, dist_types_),
        tensor_local(device,
                     this->tensor_local_shape,
                     init_n(dim_types_, this->tensor_local_shape.size()),
                     alloc_type,
                     stream.value_or(ComputeStream{device}))
  {}

  DistTensor(Device device,
             ProcessorGrid grid_,
             TensorAllocation alloc_type = StrictAlloc,
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
      : BaseDistTensor<T>(ViewType::Mutable,
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
      : BaseDistTensor<T>(ViewType::Const,
                          global_shape_,
                          dim_types_,
                          grid_,
                          dist_types_,
                          local_shape_),
        tensor_local(
            device, buffer, local_shape_, dim_types_, local_strides_, stream)
  {}

  Device get_device() const H2_NOEXCEPT override
  {
    return tensor_local.get_device();
  }

  void empty() override
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

  void resize(const ShapeTuple& new_shape) override
  {
    resize(new_shape, this->tensor_dim_types, this->tensor_dist_types);
  }

  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types) override
  {
    resize(new_shape, new_dim_types, this->tensor_dist_types);
  }

  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types,
              const DistributionTypeTuple& new_dist_types) override
  {
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_shape.size() == this->tensor_grid.ndim(),
                     "Cannot change the number of dimensions when resizing a "
                     "distributed tensor");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "Shape and dimension types must be the same size");
    H2_ASSERT_ALWAYS(new_dist_types.size() == new_shape.size(),
                     "Shape and distribution types must be the same size");
    this->tensor_shape = new_shape;
    this->tensor_local_shape = internal::get_local_shape(
        new_shape, this->tensor_grid, new_dist_types);
    this->tensor_dim_types = new_dim_types;
    this->tensor_dist_types = new_dist_types;
    tensor_local.resize(this->tensor_local_shape,
                        init_n(new_dim_types, this->tensor_local_shape.size()));
  }

  T* data() override
  {
    return tensor_local.data();
  }

  const T* data() const override
  {
    return tensor_local.data();
  }

  const T* const_data() const override
  {
    return tensor_local.const_data();
  }

  Tensor<T>& local_tensor() override
  {
    return tensor_local;
  }

  const Tensor<T>& local_tensor() const override
  {
    return tensor_local;
  }

  const Tensor<T>& const_local_tensor() const override
  {
    return tensor_local;
  }

  void ensure() override
  {
    ensure(TensorAttemptRecovery);
  }

  void ensure(tensor_no_recovery_t) override
  {
    tensor_local.ensure(TensorNoRecovery);
  }

  void ensure(tensor_attempt_recovery_t) override
  {
    tensor_local.ensure(TensorAttemptRecovery);
  }

  void release() override
  {
    tensor_local.release();
  }

  DistTensor<T>* view() override
  {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  DistTensor<T>* view() const override
  {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  DistTensor<T>* view(const IndexRangeTuple& coords) override
  {
    return make_view(coords, ViewType::Mutable);
  }

  DistTensor<T>* view(const IndexRangeTuple& coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  DistTensor<T>* operator()(const IndexRangeTuple& coords) override
  {
    return view(coords);
  }

  void unview() override
  {
    H2_ASSERT_DEBUG(this->is_view(), "Must be a view to unview");
    empty();  // Emptying a view is equivalent to unviewing.
  }

  DistTensor<T>* const_view() const override
  {
    return const_view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  DistTensor<T>* const_view(const IndexRangeTuple& coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  DistTensor<T>* operator()(const IndexRangeTuple& coords) const override
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

  /** Private constructor for views. */
  DistTensor(ViewType view_type_,
             const Tensor<T>& orig_tensor_local_,
             const ShapeTuple& local_shape_,
             const IndexRangeTuple& local_coords_,
             const ShapeTuple& shape_,
             const DimensionTypeTuple& dim_types_,
             ProcessorGrid grid_,
             const DistributionTypeTuple& dist_types_)
      : BaseDistTensor<T>(
          view_type_, shape_, dim_types_, grid_, dist_types_, local_shape_),
        tensor_local(view_type_,
                     orig_tensor_local_.tensor_memory,
                     local_shape_,
                     // Local shape may be empty depending on the view
                     // coordinates and distribution (e.g., Single).
                     local_shape_.is_empty() ? DimensionTypeTuple{}
                                             : dim_types_,
                     local_coords_)
  {}

  /** Helper for constructing views. */
  DistTensor<T>* make_view(IndexRangeTuple index_range,
                           ViewType view_type) const
  {
    H2_ASSERT_ALWAYS(is_index_range_contained(index_range, this->tensor_shape),
                     "Cannot construct an out-of-range view");
    H2_ASSERT_ALWAYS(
        !any_of(index_range,
                [](const IndexRangeTuple::type& x) { return x.is_scalar(); }),
        "Scalar indices are not permitted in global views");

    // We have three cases:
    // 1. The indices are empty, so we have a globally empty view.
    // 2. The indices result in a valid global view, but an empty local
    // tensor.
    // 3. The indices result in a valid global view and we have local
    // data.

    if (is_index_range_empty(index_range))
    {
      // Globally empty view.
      return new DistTensor<T>(
          view_type,
          tensor_local,
          ShapeTuple{},
          IndexRangeTuple{},
          ShapeTuple{},
          DimensionTypeTuple{},
          this->tensor_grid,
          DistributionTypeTuple{});
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
      return new DistTensor<T>(view_type,
                               tensor_local,
                               ShapeTuple{},
                               IndexRangeTuple{},
                               view_global_shape,
                               this->tensor_dim_types,
                               this->tensor_grid,
                               this->tensor_dist_types);
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
    return new DistTensor<T>(view_type,
                             tensor_local,
                             view_local_shape,
                             local_indices,
                             view_global_shape,
                             this->tensor_dim_types,
                             this->tensor_grid,
                             this->tensor_dist_types);
  }

};

}  // namespace h2
