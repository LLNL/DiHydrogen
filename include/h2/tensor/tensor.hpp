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

#include "h2/core/types.hpp"
#include "h2/tensor/strided_memory.hpp"
#include "h2/tensor/tensor_base.hpp"
#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/tensor_utils.hpp"
#include "h2/utils/passkey.hpp"

#include <memory>
#include <optional>

namespace h2
{

// Forward-declaration:
template <typename T>
class DistTensor;

/** Tensor class for arbitrary types and devices. */
template <typename T>
class Tensor : public BaseTensor
{
public:
  using value_type = T;

  static_assert(IsH2StorageType_v<T>,
                "Cannot create a tensor with a non-storage type");

  Tensor(Device device,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         TensorAllocationStrategy alloc_type = StrictAlloc,
         std::optional<ComputeStream> const stream = std::nullopt)
    : BaseTensor(shape_, dim_types_),
      tensor_memory(device,
                    shape_,
                    alloc_type == LazyAlloc,
                    stream.value_or(ComputeStream{device}))
  {}

  Tensor(Device device,
         TensorAllocationStrategy alloc_type = StrictAlloc,
         std::optional<ComputeStream> const stream = std::nullopt)
    : Tensor(device, ShapeTuple(), DimensionTypeTuple(), alloc_type, stream)
  {}

  Tensor(Device device,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         StrideTuple const& strides_,
         TensorAllocationStrategy alloc_type = StrictAlloc,
         std::optional<ComputeStream> const stream = std::nullopt)
    : BaseTensor(shape_, dim_types_),
      tensor_memory(device,
                    shape_,
                    strides_,
                    alloc_type == LazyAlloc,
                    stream.value_or(ComputeStream{device}))
  {}

  Tensor(Device device,
         T* buffer,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         StrideTuple const& strides_,
         ComputeStream const& stream)
    : BaseTensor(ViewType::Mutable, shape_, dim_types_),
      tensor_memory(device, buffer, shape_, strides_, stream)
  {}

  Tensor(Device device,
         T const* buffer,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         StrideTuple const& strides_,
         ComputeStream const& stream)
    : BaseTensor(ViewType::Const, shape_, dim_types_),
      tensor_memory(device, const_cast<T*>(buffer), shape_, strides_, stream)
  {}

  /** Internal constructor for views. */
  Tensor(ViewType view_type_,
         StridedMemory<T> const& mem_,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         IndexRangeTuple const& coords,
         Passkey2<Tensor<T>, DistTensor<T>>)
    : BaseTensor(view_type_, shape_, dim_types_), tensor_memory(mem_, coords)
  {}

  /**
   * Internal constructor for views from different devices.
   *
   * Not protected by a passkey because we use it from a free function.
   */
  Tensor(Tensor<T>& other, Device new_device, ComputeStream const& new_stream)
    : BaseTensor(ViewType::Mutable, other.shape(), other.dim_types()),
      tensor_memory(other.tensor_memory, new_device, new_stream)
  {}

  Tensor(Tensor<T> const& other,
         Device new_device,
         ComputeStream const& new_stream)
    : BaseTensor(ViewType::Const, other.shape(), other.dim_types()),
      tensor_memory(other.tensor_memory, new_device, new_stream)
  {}

  /** Internal constructor for cloning. */
  Tensor(StridedMemory<T> const& mem_,
         ShapeTuple const& shape_,
         DimensionTypeTuple const& dim_types_,
         Passkey2<Tensor<T>, DistTensor<T>>)
    : BaseTensor(shape_, dim_types_), tensor_memory(mem_)
  {}

  virtual ~Tensor() = default;

  /**
   * Disable copy construction.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  Tensor(Tensor const&) = delete;

  /**
   * Disable copy assignment.
   *
   * Using it leads to ambiguity in mutable vs const views. Create a
   * view or copy explicitly instead.
   */
  Tensor& operator=(Tensor const&) = delete;

  /** Move construction */
  Tensor(Tensor&&) = default;

  /** Move assignment */
  Tensor& operator=(Tensor&&) = default;

  /**
   * Return an exact copy of this tensor.
   *
   * The new tensor will always have distinct memory that it manages
   * (i.e., unlike a view, memory is not shared). A clone is never a
   * view, regardless of whether the original tensor is. If the tensor
   * is viewing an external memory buffer, the new tensor will have a
   * copy of the buffer, and will manage it directly. If the tensor is
   * lazy, the clone will be as well; it will not be able to recover
   * memory from existing views, if any.
   */
  std::unique_ptr<Tensor<T>> clone() const
  {
    // Abuses the view constructor.
    StridedMemory<T> cloned_mem = tensor_memory.clone();
    return std::make_unique<Tensor<T>>(
      cloned_mem, shape(), dim_types(), Passkey<Tensor<T>>{});
  }

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

  TypeInfo get_type_info() const H2_NOEXCEPT override
  {
    return get_h2_type<T>();
  }

  StrideTuple strides() const H2_NOEXCEPT override
  {
    return tensor_memory.strides();
  }

  typename StrideTuple::type
  stride(typename StrideTuple::size_type i) const H2_NOEXCEPT override
  {
    return tensor_memory.strides()[i];
  }

  bool is_contiguous() const H2_NOEXCEPT override
  {
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
    if (this->is_view())
    {
      this->tensor_view_type = ViewType::None;
    }
  }

  void resize(ShapeTuple const& new_shape) override
  {
    H2_ASSERT_ALWAYS(new_shape.size() <= this->tensor_shape.size(),
                     "Must provide dimension types to resize larger");
    resize(new_shape, init_n(this->tensor_dim_types, new_shape.size()));
  }

  void resize(ShapeTuple const& new_shape,
              DimensionTypeTuple const& new_dim_types) override
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

  void resize(ShapeTuple const& new_shape,
              DimensionTypeTuple const& new_dim_types,
              StrideTuple const& new_strides) override
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
  T* data()
  {
    if (this->tensor_view_type == ViewType::Const)
    {
      throw H2Exception("Cannot access non-const buffer of const view");
    }
    ensure();
    return tensor_memory.data();
  }

  /** Return a raw constant pointer to the underlying storage. */
  T const* data() const { return tensor_memory.const_data(); }

  /** Return a raw constant pointer to the underlying storage. */
  T const* const_data() const { return tensor_memory.const_data(); }

  void* storage_data() override { return static_cast<void*>(data()); }

  void const* storage_data() const override
  {
    return static_cast<void const*>(const_data());
  }

  void const* const_storage_data() const override
  {
    return static_cast<void const*>(const_data());
  }

  void ensure() override { ensure(TensorAttemptRecovery); }

  void ensure(tensor_no_recovery_t) override { tensor_memory.ensure(false); }

  void ensure(tensor_attempt_recovery_t) override
  {
    tensor_memory.ensure(true);
  }

  void release() override { tensor_memory.release(); }

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
  std::unique_ptr<Tensor<T>> contiguous()
  {
    if (is_contiguous())
    {
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
  std::unique_ptr<Tensor<T>> view()
  {
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
   * fully empty, the view will be empty.
   *
   * If dimensions in `coords` are given as scalars, these dimensions
   * are eliminated from the tensor. If all dimensions are eliminated,
   * i.e., you access a specific element, the resulting view will have
   * one dimension with dimension-type `Scalar`.
   */
  std::unique_ptr<Tensor<T>> view(IndexRangeTuple const& coords)
  {
    return make_view(coords, ViewType::Mutable);
  }

  /**
   * Return a constant view of a subtensor of this tensor.
   */
  std::unique_ptr<Tensor<T>> view(IndexRangeTuple const& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for view(coords). */
  std::unique_ptr<Tensor<T>> operator()(IndexRangeTuple const& coords)
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
  std::unique_ptr<Tensor<T>> const_view() const
  {
    return const_view(IndexRangeTuple(
      TuplePad<IndexRangeTuple>(this->tensor_shape.size(), ALL)));
  }

  /** Return a constant view of a subtensor of this tensor. */
  std::unique_ptr<Tensor<T>> const_view(IndexRangeTuple const& coords) const
  {
    return make_view(coords, ViewType::Const);
  }

  /** Convenience wrapper for const_view(coords). */
  std::unique_ptr<Tensor<T>> operator()(IndexRangeTuple const& coords) const
  {
    return const_view(coords);
  }

  /** Return a pointer to the tensor at a particular coordinate. */
  T* get(ScalarIndexTuple const& coords)
  {
    H2_ASSERT_DEBUG(is_index_in_shape(coords, shape()),
                    "Cannot get index ",
                    coords,
                    " in tensor with shape ",
                    shape());
    if (this->tensor_view_type == ViewType::Const)
    {
      throw H2Exception("Cannot access non-const buffer of const view");
    }
    return tensor_memory.get(coords);
  }

  /**
   * Return a constant pointer to the tensor at a particular coordinate.
   */
  T const* get(ScalarIndexTuple const& coords) const
  {
    H2_ASSERT_DEBUG(is_index_in_shape(coords, shape()),
                    "Cannot get index ",
                    coords,
                    " in tensor with shape ",
                    shape());
    return tensor_memory.get(coords);
  }

  /**
   * Return a constant pointer to the tensor at a particular coordinate.
   */
  T const* const_get(ScalarIndexTuple const& coords) const
  {
    H2_ASSERT_DEBUG(is_index_in_shape(coords, shape()),
                    "Cannot get index ",
                    coords,
                    " in tensor with shape ",
                    shape());
    return tensor_memory.const_get(coords);
  }

  ComputeStream get_stream() const H2_NOEXCEPT override
  {
    return tensor_memory.get_stream();
  }

  void set_stream(ComputeStream const& stream) override
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
  std::unique_ptr<Tensor<T>> make_view(IndexRangeTuple const& coords,
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
      filtered_dim_types =
        filter_index(this->tensor_dim_types, [&](IndexRangeTuple::size_type i) {
          return !coords[i].is_scalar();
        });
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
