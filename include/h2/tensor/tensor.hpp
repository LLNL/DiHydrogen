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

namespace h2
{

// Forward declaration:
template <typename T, Device Dev> class DistTensor;

/** Tensor class for arbitrary types and devices. */
template <typename T, Device Dev>
class Tensor : public BaseTensor<T> {
public:
  using value_type = T;
  static constexpr Device device = Dev;

  Tensor(ShapeTuple shape_,
         DimensionTypeTuple dim_types_,
         const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
      : Tensor(shape_, dim_types_, UnlazyAlloc, sync)
  {}

  Tensor(ShapeTuple shape_,
         DimensionTypeTuple dim_types_,
         lazy_alloc_t,
         const SyncInfo<Dev>& sync = SyncInfo<Dev>{}) :
    BaseTensor<T>(shape_, dim_types_),
    tensor_memory(shape_, true, sync)
  {}

  Tensor(ShapeTuple shape_,
         DimensionTypeTuple dim_types_,
         unlazy_alloc_t,
         const SyncInfo<Dev>& sync = SyncInfo<Dev>{}) :
    BaseTensor<T>(shape_, dim_types_),
    tensor_memory(shape_, false, sync)
  {}

  Tensor(const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : Tensor(ShapeTuple(), DimensionTypeTuple(), UnlazyAlloc, sync) {}

  Tensor(lazy_alloc_t, const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : Tensor(ShapeTuple(), DimensionTypeTuple(), LazyAlloc, sync) {}

  Tensor(unlazy_alloc_t, const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : Tensor(ShapeTuple(), DimensionTypeTuple(), UnlazyAlloc, sync) {}

  Tensor(T* buffer,
         ShapeTuple shape_,
         DimensionTypeTuple dim_types_,
         StrideTuple strides_,
         const SyncInfo<Dev>& sync = SyncInfo<Dev>{}) :
    BaseTensor<T>(ViewType::Mutable, shape_, dim_types_),
    tensor_memory(buffer, shape_, strides_, sync)
  {}

  Tensor(const T* buffer,
         ShapeTuple shape_,
         DimensionTypeTuple dim_types_,
         StrideTuple strides_,
         const SyncInfo<Dev>& sync = SyncInfo<Dev>{}) :
    BaseTensor<T>(ViewType::Const, shape_, dim_types_),
    tensor_memory(const_cast<T*>(buffer), shape_, strides_, sync)
  {}

  StrideTuple strides() const H2_NOEXCEPT override {
    return tensor_memory.strides();
  }

  typename StrideTuple::type stride(typename StrideTuple::size_type i) const H2_NOEXCEPT override {
    return tensor_memory.strides()[i];
  }

  bool is_contiguous() const H2_NOEXCEPT override {
    return are_strides_contiguous(this->tensor_shape, tensor_memory.strides());
  }

  Device get_device() const H2_NOEXCEPT override { return device; }

  void empty() override
  {
    auto sync = tensor_memory.get_sync_info();
    tensor_memory = StridedMemory<T, Dev>(tensor_memory.is_lazy(), sync);
    this->tensor_shape = ShapeTuple();
    this->tensor_dim_types = DimensionTypeTuple();
    if (this->is_view()) {
      this->tensor_view_type = ViewType::None;
    }
  }

  void resize(ShapeTuple new_shape) override {
    if (this->is_view()) {
      throw H2Exception("Cannot resize a view");
    }
    if (new_shape.size() > this->tensor_shape.size()) {
      throw H2Exception("Must provide dimension types to resize larger");
    }
    auto sync = tensor_memory.get_sync_info();
    tensor_memory = StridedMemory<T, Dev>(
      new_shape, tensor_memory.is_lazy(), sync);
    this->tensor_shape = new_shape;
    this->tensor_dim_types.set_size(new_shape.size());
  }

  void resize(ShapeTuple new_shape, DimensionTypeTuple new_dim_types) override {
    if (this->is_view()) {
      throw H2Exception("Cannot resize a view");
    }
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "New shape and dimension types must have the same size");
    auto sync = tensor_memory.get_sync_info();
    tensor_memory = StridedMemory<T, Dev>(
      new_shape, tensor_memory.is_lazy(), sync);
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

  Tensor<T, Dev>* contiguous() override {
    if (is_contiguous()) {
      return view();
    }
    throw H2Exception("contiguous() not implemented");
  }

  Tensor<T, Dev>* view() override {
    return view(CoordTuple(TuplePad<CoordTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T, Dev>* view() const override
  {
    return view(CoordTuple(TuplePad<CoordTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T, Dev>* view(CoordTuple coords) override
  {
    return make_view(coords, ViewType::Mutable);
  }

  Tensor<T, Dev>* view(CoordTuple coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  Tensor<T, Dev>* operator()(CoordTuple coords) override
  {
    return view(coords);
  }

  void unview() override {
    H2_ASSERT_DEBUG(this->is_view(), "Must be a view to unview");
    empty();  // Emptying a view is equivalent to unviewing.
  }

  Tensor<T, Dev>* const_view() const override {
    return const_view(CoordTuple(TuplePad<CoordTuple>(this->tensor_shape.size(), ALL)));
  }

  Tensor<T, Dev>* const_view(CoordTuple coords) const override
  {
    return make_view(coords, ViewType::Const);
  }

  Tensor<T, Dev>* operator()(CoordTuple coords) const override {
    return const_view(coords);
  }

  T* get(SingleCoordTuple coords) override
  {
    return tensor_memory.get(coords);
  }

  const T* get(SingleCoordTuple coords) const override
  {
    return tensor_memory.get(coords);
  }

  SyncInfo<Dev> get_sync_info() const H2_NOEXCEPT
  {
    return tensor_memory.get_sync_info();
  }

  void set_sync_info(const SyncInfo<Dev>& sync)
  {
    if (this->is_view())
    {
      tensor_memory.set_sync_info(sync);
    }
    else
    {
      tensor_memory.set_sync_info(sync, true);
    }
  }

  bool is_lazy() const H2_NOEXCEPT
  {
    return tensor_memory.is_lazy();
  }

private:
  /** Underlying memory buffer for the tensor. */
  StridedMemory<T, Dev> tensor_memory;

  /** Private constructor for views. */
  Tensor(ViewType view_type_, const StridedMemory<T, Dev>& mem_,
         ShapeTuple shape_, DimensionTypeTuple dim_types_, CoordTuple coords) :
    BaseTensor<T>(view_type_, shape_, dim_types_),
    tensor_memory(mem_, coords)
  {}

  /** Helper for constructing views. */
  Tensor<T, Dev>* make_view(CoordTuple coords, ViewType view_type) const
  {
    if (!is_shape_contained(coords, this->tensor_shape))
    {
      throw H2Exception("Attempting to construct an out-of-range view");
    }
    if (is_range_empty(coords))
    {
      return new Tensor<T, Dev>(view_type,
                                tensor_memory,
                                ShapeTuple{},
                                DimensionTypeTuple{},
                                CoordTuple{});
    }
    ShapeTuple view_shape = get_range_shape(coords, this->tensor_shape);
    return new Tensor<T, Dev>(
        view_type,
        tensor_memory,
        get_range_shape(coords, this->tensor_shape),
        filter_by_trivial(coords, this->tensor_dim_types),
        coords);
  }
};

}  // namespace h2
