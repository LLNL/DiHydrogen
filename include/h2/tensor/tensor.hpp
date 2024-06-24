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

#include <memory>
#include <optional>

#include "h2/tensor/tensor_base.hpp"
#include "h2/tensor/strided_memory.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"
#include "h2/utils/passkey.hpp"

namespace h2
{

// Forward-declaration:
template <typename T> class DistTensor;

/** Tensor class for arbitrary types and devices. */
template <typename T>
class Tensor : public BaseTensor {
public:
  using value_type = T;

  Tensor(Device device,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         TensorAllocationStrategy alloc_type = StrictAlloc,
         const std::optional<ComputeStream> stream = std::nullopt)
      : BaseTensor(shape_, dim_types_),
        tensor_memory(device,
                      shape_,
                      alloc_type == LazyAlloc,
                      stream.value_or(ComputeStream{device}))
  {}

  Tensor(Device device,
         TensorAllocationStrategy alloc_type = StrictAlloc,
         const std::optional<ComputeStream> stream = std::nullopt)
      : Tensor(device, ShapeTuple(), DimensionTypeTuple(), alloc_type, stream)
  {}

  Tensor(Device device,
         T* buffer,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const StrideTuple& strides_,
         const ComputeStream& stream) :
    BaseTensor(ViewType::Mutable, shape_, dim_types_),
    tensor_memory(device, buffer, shape_, strides_, stream)
  {}

  Tensor(Device device,
         const T* buffer,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const StrideTuple& strides_,
         const ComputeStream& stream) :
    BaseTensor(ViewType::Const, shape_, dim_types_),
    tensor_memory(device, const_cast<T*>(buffer), shape_, strides_, stream)
  {}

  /** Internal constructor for views. */
  Tensor(ViewType view_type_,
         const StridedMemory<T>& mem_,
         const ShapeTuple& shape_,
         const DimensionTypeTuple& dim_types_,
         const IndexRangeTuple& coords,
         Passkey2<Tensor<T>, DistTensor<T>>)
      :
    BaseTensor(view_type_, shape_, dim_types_),
    tensor_memory(mem_, coords)
  {}

  /**
   * Internal constructor for views from different devices.
   *
   * Not protected by a passkey because we use it from a free function.
   */
  Tensor(Tensor<T>& other, Device new_device, const ComputeStream& new_stream)
      : BaseTensor(ViewType::Mutable, other.shape(), other.dim_types()),
        tensor_memory(other.tensor_memory, new_device, new_stream)
  {}

  virtual ~Tensor() = default;

  /**
   * Disable copy construction.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  Tensor(const Tensor&) = delete;

  /**
   * Disable copy assignment.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  Tensor& operator=(const Tensor&) = delete;

  /** Move construction */
  Tensor(Tensor&&) = default;

  /** Move assignment */
  Tensor& operator=(Tensor&&) = default;

  /** Output a short description of the tensor. */
  void short_describe(std::ostream& os) const override
  {
    os << "Tensor<" << TypeName<T>() << ", " << get_device() << ">(";
    if (is_view())
    {
      os << get_view_type() << " of ";
    }
    for (ShapeTuple::size_type i = 0; i < ndim(); ++i)
    {
      os << dim_type(i) << ":" << shape(i);
      if (i < ndim() - 1)
      {
        os << " x ";
      }
    }
    os << ")";
  }

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

  /**
   * Clear the tensor and reset it to empty.
   *
   * If this is a view, this is equivalent to `unview`.
   */
  void empty()
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

  /**
   * Resize the tensor to a new shape, keeping dimension types the same.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape)
  {
    H2_ASSERT_ALWAYS(new_shape.size() <= this->tensor_shape.size(),
                     "Must provide dimension types to resize larger");
    resize(new_shape, init_n(this->tensor_dim_types, new_shape.size()));
  }

  /**
   * Resize the tensor to a new shape, also changing dimension types.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types)
  {
    // We do not call the resize-with-new-strides version so we do not
    // have to compute the strides manually.
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "New shape (",
                     new_shape,
                     ") and dimension types (",
                     new_dim_types,
                     ") must have the same size");
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

  /**
   * Resize the tensor to a new shape, also changing dimension types
   * and specifying new strides.
   *
   * It is an error to call this on a view.
   */
  void resize(const ShapeTuple& new_shape,
              const DimensionTypeTuple& new_dim_types,
              const StrideTuple& new_strides)
  {
    H2_ASSERT_ALWAYS(!this->is_view(), "Cannot resize a view");
    H2_ASSERT_ALWAYS(new_dim_types.size() == new_shape.size(),
                     "New shape (",
                     new_shape,
                     ") and dimension types (",
                     new_dim_types,
                     ") must have the same size");
    H2_ASSERT_ALWAYS(new_strides.is_empty()
                         || new_strides.size() == new_shape.size(),
                     "New shape (",
                     new_shape,
                     ") and strides (",
                     new_strides,
                     ") must have the same size");
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

  /**
   * Return a raw pointer to the underlying storage.
   *
   * @note Remember to account for the strides when accessing this.
   */
  T* data() {
    if (this->tensor_view_type == ViewType::Const) {
      throw H2Exception("Cannot access non-const buffer of const view");
    }
    ensure();
    return tensor_memory.data();
  }

  /** Return a raw constant pointer to the underlying storage. */
  const T* data() const
  {
    return tensor_memory.const_data();
  }

  /** Return a raw constant pointer to the underlying storage. */
  const T* const_data() const {
    return tensor_memory.const_data();
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
    tensor_memory.ensure(false);
  }

  /**
   * Ensure memory is backing this tensor, allocating if necessary.
   *
   * This attempts to reuse existing memory from still-extant views of
   * this tensor.
   */
  void ensure(tensor_attempt_recovery_t)
  {
    tensor_memory.ensure(true);
  }

  /**
   * Release memory associated with this tensor.
   *
   * Note that if there are views, memory may not be deallocated
   * immediately.
   */
  void release()
  {
    tensor_memory.release();
  }

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
  std::unique_ptr<Tensor<T>> contiguous() {
    if (is_contiguous()) {
      return view();
    }
    throw H2Exception("contiguous() not implemented");
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
  std::unique_ptr<Tensor<T>> view() {
    return view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  /** Return a constant view of this tensor. */
  std::unique_ptr<Tensor<T>> view() const
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
   * fully empty, the view iwll be empty.
   *
   * If dimensions in `coords` are given as scalars, these dimensions
   * are eliminated from the tensor. If all dimensions are eliminated,
   * i.e., you access a specific element, the resulting view will have
   * one dimension with dimension-type `Scalar`.
   */
  std::unique_ptr<Tensor<T>> view(const IndexRangeTuple& coords)
  {
    return make_view(coords, ViewType::Mutable);
  }

  /**
   * Return a constant view of a subtensor of this tensor.
   */
  std::unique_ptr<Tensor<T>> view(const IndexRangeTuple& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for view(coords). */
  std::unique_ptr<Tensor<T>> operator()(const IndexRangeTuple& coords)
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
  void unview() {
    H2_ASSERT_DEBUG(this->is_view(), "Must be a view to unview");
    empty();  // Emptying a view is equivalent to unviewing.
  }

  /** Return a constant view of this tensor. */
  std::unique_ptr<Tensor<T>> const_view() const {
    return const_view(IndexRangeTuple(
        TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  /** Return a constant view of a subtensor of this tensor. */
  std::unique_ptr<Tensor<T>> const_view(const IndexRangeTuple& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for const_view(coords). */
  std::unique_ptr<Tensor<T>> operator()(const IndexRangeTuple& coords) const {
    return const_view(coords);
  }

  /** Return a pointer to the tensor at a particular coordinate. */
  T* get(const ScalarIndexTuple& coords)
  {
    return tensor_memory.get(coords);
  }

  /**
   * Return a constant pointer to the tensor at a particular coordinate.
   */
  const T* get(const ScalarIndexTuple& coords) const
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

  /** Helper for constructing views. */
  std::unique_ptr<Tensor<T>> make_view(const IndexRangeTuple& coords,
                                       ViewType view_type) const
  {
    if (!is_index_range_contained(coords, this->tensor_shape))
    {
      throw H2Exception("Attempting to construct an out-of-range view, ",
                        coords,
                        " is not in ",
                        this->tensor_shape);
    }
    // We need an explicit check here because specific coordinates may
    // be empty. We can handle empty IndexRangeTuples, but not empty
    // IndexRanges.
    if (is_index_range_empty(coords))
    {
      return std::make_unique<Tensor<T>>(view_type,
                                         tensor_memory,
                                         ShapeTuple{},
                                         DimensionTypeTuple{},
                                         IndexRangeTuple{},
                                         Passkey<Tensor<T>>{});
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
    return std::make_unique<Tensor<T>>(view_type,
                                       tensor_memory,
                                       view_shape,
                                       filtered_dim_types,
                                       coords,
                                       Passkey<Tensor<T>>{});
  }

  // DistTensor needs to poke in here for some view stuff.
  friend class DistTensor<T>;
};

}  // namespace h2
