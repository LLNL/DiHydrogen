////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Local tensors that live on a device.
 */

#include "h2/tensor/tensor_base.hpp"
#include "h2/tensor/strided_memory.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"

namespace h2
{

// Forward-declaration:
template <typename T> class DistTensor;

/** Tensor class for arbitrary types and devices. */
template <typename T>
class Tensor : public BaseTensor<T> {
public:
  using value_type = T;

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const ComputeStream& stream)
    : Tensor(device, shape_, dim_types_, StrictAlloc, stream)
  {}

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_)
    : Tensor(device, shape_, dim_types_, ComputeStream{device}) {}

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         lazy_alloc_t,
         const ComputeStream& stream) :
    BaseTensor<T>(shape_, dim_types_),
    tensor_memory(device, shape_, true, stream)
  {}

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         lazy_alloc_t)
      : Tensor(device, shape_, dim_types_, LazyAlloc, ComputeStream{device})
  {}

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         strict_alloc_t,
         const ComputeStream& stream) :
    BaseTensor<T>(shape_, dim_types_),
    tensor_memory(device, shape_, false, stream)
  {}

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         strict_alloc_t)
      : Tensor(device, shape_, dim_types_, StrictAlloc, ComputeStream{device})
  {}

  Tensor(Device device, const ComputeStream& stream)
      : Tensor(device, ShapeTuple(), DimensionTypeTuple(), StrictAlloc, stream)
  {}

  Tensor(Device device) : Tensor(device, ComputeStream{device}) {}

  Tensor(Device device, lazy_alloc_t, const ComputeStream& stream)
      : Tensor(device, ShapeTuple(), DimensionTypeTuple(), LazyAlloc, stream)
  {}

  Tensor(Device device, lazy_alloc_t)
      : Tensor(device, LazyAlloc, ComputeStream{device})
  {}

  Tensor(Device device, strict_alloc_t, const ComputeStream& stream)
      : Tensor(device, ShapeTuple(), DimensionTypeTuple(), StrictAlloc, stream)
  {}

  Tensor(Device device, strict_alloc_t)
      : Tensor(device, StrictAlloc, ComputeStream{device})
  {}

  Tensor(Device device,
         T* buffer,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const StrideTuple& strides_,
         const ComputeStream& stream) :
    BaseTensor<T>(ViewType::Mutable, shape_, dim_types_),
    tensor_memory(device, buffer, shape_, strides_, stream)
  {}

  Tensor(Device device,
         const T* buffer,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const StrideTuple& strides_,
         const ComputeStream& stream) :
    BaseTensor<T>(ViewType::Const, shape_, dim_types_),
    tensor_memory(device, const_cast<T*>(buffer), shape_, strides_, stream)
  {}

  StrideTuple strides() const H2_NOEXCEPT override {
    return tensor_memory.strides();
  }

  typename StrideTuple::type
  stride(typename StrideTuple::size_type i) const H2_NOEXCEPT override
  {
    return tensor_memory.strides()[i];
  }

  bool is_contiguous() const H2_NOEXCEPT override {
    return are_strides_contiguous(this->tensor_shape, tensor_memory.strides());
  }

  Device get_device() const H2_NOEXCEPT override
  {
    return tensor_memory.get_device();
  }

  void empty() override
  {
    auto stream = tensor_memory.get_stream();
    tensor_memory =
        StridedMemory<T>(get_device(), tensor_memory.is_lazy(), stream);
    this->tensor_shape = ShapeTuple();
    this->tensor_dim_types = DimensionTypeTuple();
    if (this->is_view()) {
      this->tensor_view_type = ViewType::None;
    }
  }

  void resize(const ShapeTuple& new_shape) override
  {
    H2_ASSERT_ALWAYS(new_shape.size() <= this->tensor_shape.size(),
                     "Must provide dimension types to resize larger");
    resize(new_shape, init_n(this->tensor_dim_types, new_shape.size()));
  }

  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types) override
  {
    // We do not call the resize-with-new-strides version so we do not
    // have to compute the strides manually.
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "New shape and dimension types must have the same size");
    // Don't reallocate if we would not change the size.
    if (this->tensor_shape == new_shape)
    {
      // May still change the dimension types.
      this->tensor_dim_types = new_dim_types;
      return;
    }
    auto stream = tensor_memory.get_stream();
    tensor_memory = StridedMemory<T>(
      get_device(), new_shape, tensor_memory.is_lazy(), stream);
    this->tensor_shape = new_shape;
    this->tensor_dim_types = new_dim_types;
  }

  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types,
              const StrideTuple& new_strides) override
  {
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "New shape and dimension types must have the same size");
    H2_ASSERT_ALWAYS(new_strides.is_empty()
                         || new_strides.size() == new_shape.size(),
                     "New shape and strides must have the same size");
    // Don't reallocate if we would not change the size.
    if (this->tensor_shape == new_shape && strides() == new_strides)
    {
      // May still change the dimension types.
      this->tensor_dim_types = new_dim_types;
      return;
    }
    auto stream = tensor_memory.get_stream();
    tensor_memory = StridedMemory<T>(
      get_device(), new_shape, new_strides, tensor_memory.is_lazy(), stream);
    this->tensor_shape = new_shape;
    this->tensor_dim_types = new_dim_types;
  }

  T* data() override {
    if (this->tensor_view_type == ViewType::Const) {
      throw H2Exception("Cannot access non-const buffer of const view");
    }
    ensure();
    return tensor_memory.data();
  }

  const T* data() const override
  {
    return tensor_memory.const_data();
  }

  const T* const_data() const override {
    return tensor_memory.const_data();
  }

  void ensure() override
  {
    ensure(TensorAttemptRecovery);
  }

  void ensure(tensor_no_recovery_t) override
  {
    tensor_memory.ensure(false);
  }

  void ensure(tensor_attempt_recovery_t) override
  {
    tensor_memory.ensure(true);
  }

  void release() override
  {
    tensor_memory.release();
  }

  Tensor<T>* contiguous() override {
    if (is_contiguous()) {
      return view();
    }
    throw H2Exception("contiguous() not implemented");
  }

  Tensor<T>* view() override {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T>* view() const override
  {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T>* view(const IndexRangeTuple& coords) override
  {
    return make_view(coords, ViewType::Mutable);
  }

  Tensor<T>* view(const IndexRangeTuple& coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  Tensor<T>* operator()(const IndexRangeTuple& coords) override
  {
    return view(coords);
  }

  void unview() override {
    H2_ASSERT_DEBUG(this->is_view(), "Must be a view to unview");
    empty();  // Emptying a view is equivalent to unviewing.
  }

  Tensor<T>* const_view() const override {
    return const_view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T>* const_view(const IndexRangeTuple& coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  Tensor<T>* operator()(const IndexRangeTuple& coords) const override {
    return const_view(coords);
  }

  T* get(const ScalarIndexTuple& coords) override
  {
    return tensor_memory.get(coords);
  }

  const T* get(const ScalarIndexTuple& coords) const override
  {
    return tensor_memory.get(coords);
  }

  ComputeStream get_stream() const H2_NOEXCEPT
  {
    return tensor_memory.get_stream();
  }

  void set_stream(const ComputeStream& stream)
  {
    if (this->is_view())
    {
      tensor_memory.set_stream(stream);
    }
    else
    {
      tensor_memory.set_stream(stream, true);
    }
  }

  bool is_lazy() const H2_NOEXCEPT { return tensor_memory.is_lazy(); }

private:
  /** Underlying memory buffer for the tensor. */
  StridedMemory<T> tensor_memory;

  /** Private constructor for views. */
  Tensor(ViewType view_type_,
         const StridedMemory<T>& mem_,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const IndexRangeTuple& coords)
      :
    BaseTensor<T>(view_type_, shape_, dim_types_),
    tensor_memory(mem_, coords)
  {}

  /** Helper for constructing views. */
  Tensor<T>* make_view(const IndexRangeTuple& coords,
                       ViewType view_type) const
  {
    if (!is_index_range_contained(coords, this->tensor_shape))
    {
      throw H2Exception("Attempting to construct an out-of-range view");
    }
    // We need an explicit check here because specific coordinates may
    // be empty. We can handle empty IndexRangeTuples, but not empty
    // IndexRanges.
    if (is_index_range_empty(coords))
    {
      return new Tensor<T>(view_type,
                           tensor_memory,
                           ShapeTuple{},
                           DimensionTypeTuple{},
                           IndexRangeTuple{});
    }
    ShapeTuple view_shape = get_index_range_shape(coords, this->tensor_shape);
    // Eliminate dimension types from dimensions that have been
    // eliminated.
    // If we would eliminate all dimensions (i.e., use scalars for all
    // coordinates), decay to a shape of 1 with a Scalar dimension.
    DimensionTypeTuple filtered_dim_types;
    if (!coords.is_empty() && view_shape.is_empty())
    {
      view_shape = ShapeTuple(1);
      filtered_dim_types = DimensionTypeTuple(DimensionType::Scalar);
    }
    else
    {
      filtered_dim_types = filter_index(
          this->tensor_dim_types,
          [&](IndexRangeTuple::size_type i) { return !coords[i].is_scalar(); });
    }
    return new Tensor<T>(view_type,
                         tensor_memory,
                         view_shape,
                         filtered_dim_types,
                         coords);
  }

  // DistTensor needs to poke in here for some view stuff.
  friend class DistTensor<T>;
};

}  // namespace h2
