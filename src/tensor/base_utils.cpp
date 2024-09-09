////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/base_utils.hpp"

#include "h2/core/dispatch.hpp"
#include "h2/utils/unique_ptr_cast.hpp"

namespace h2
{
namespace base
{

namespace
{

template <typename T>
void make_tensor_impl(std::unique_ptr<BaseTensor>& ptr,
                      Device device,
                      ShapeTuple const& shape,
                      DimensionTypeTuple const& dim_types,
                      StrideTuple const& strides,
                      TensorAllocationStrategy alloc_type,
                      ComputeStream const& stream)
{
  ptr = std::make_unique<Tensor<T>>(
    device, shape, dim_types, strides, alloc_type, stream);
}

template <typename T>
void view_impl(std::unique_ptr<BaseTensor>& ptr, BaseTensor& tensor)
{
  Tensor<T>& real_tensor = static_cast<Tensor<T>&>(tensor);
  ptr = real_tensor.view();
}

template <typename T>
void view_impl(std::unique_ptr<BaseTensor>& ptr,
               BaseTensor& tensor,
               IndexRangeTuple const& coords)
{
  Tensor<T>& real_tensor = static_cast<Tensor<T>&>(tensor);
  ptr = real_tensor.view(coords);
}

template <typename T>
void const_view_impl(std::unique_ptr<BaseTensor>& ptr, BaseTensor const& tensor)
{
  Tensor<T> const& real_tensor = static_cast<Tensor<T> const&>(tensor);
  ptr = real_tensor.const_view();
}

template <typename T>
void const_view_impl(std::unique_ptr<BaseTensor>& ptr,
                     BaseTensor const& tensor,
                     IndexRangeTuple const& coords)
{
  Tensor<T> const& real_tensor = static_cast<Tensor<T> const&>(tensor);
  ptr = real_tensor.const_view(coords);
}

}  // anonymous namespace

std::unique_ptr<BaseTensor>
make_tensor(TypeInfo const& tinfo,
            Device device,
            ShapeTuple const& shape,
            DimensionTypeTuple const& dim_types,
            StrideTuple const& strides,
            TensorAllocationStrategy alloc_type,
            std::optional<ComputeStream> const stream)
{
  // H2_DISPATCH_NAME: make_tensor
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: make_tensor_impl<{T1}>("std::unique_ptr<BaseTensor>&", "Device", "const ShapeTuple&", "const DimensionTypeTuple&", "const StrideTuple&", "TensorAllocationStrategy", "const ComputeStream&")

  StrideTuple real_strides = (strides.is_empty() && !shape.is_empty())
                               ? get_contiguous_strides(shape)
                               : strides;
  ComputeStream real_stream = stream.value_or(ComputeStream{device});
  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tinfo"
  // H2_DISPATCH_ARGS: "ptr", "device", "shape", "dim_types", "real_strides", "alloc_type", "real_stream"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> view(BaseTensor& tensor)
{
  // H2_DISPATCH_NAME: view
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "BaseTensor&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> view(BaseTensor& tensor,
                                 IndexRangeTuple const& coords)
{
  // H2_DISPATCH_NAME: view_coords
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "BaseTensor&", "const IndexRangeTuple&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor", "coords"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> view(BaseTensor const& tensor)
{
  // H2_DISPATCH_NAME: const_view
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: const_view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "const BaseTensor&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> view(BaseTensor const& tensor,
                                 IndexRangeTuple const& coords)
{
  // H2_DISPATCH_NAME: const_view_coords
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: const_view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "const BaseTensor&", "const IndexRangeTuple&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor", "coords"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> const_view(BaseTensor const& tensor)
{
  // H2_DISPATCH_NAME: const_view
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: const_view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "const BaseTensor&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor"
  // H2_DO_DISPATCH

  return ptr;
}

std::unique_ptr<BaseTensor> const_view(BaseTensor const& tensor,
                                       IndexRangeTuple const& coords)
{
  // H2_DISPATCH_NAME: const_view_coords
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: const_view_impl<{T1}>("std::unique_ptr<BaseTensor>&", "const BaseTensor&", "const IndexRangeTuple&")

  std::unique_ptr<BaseTensor> ptr = nullptr;

  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS: "ptr", "tensor", "coords"
  // H2_DO_DISPATCH

  return ptr;
}

}  // namespace base
}  // namespace h2
