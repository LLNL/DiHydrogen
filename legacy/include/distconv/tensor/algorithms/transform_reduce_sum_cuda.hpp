#pragma once

#include "distconv/tensor/algorithms/reduce_sum_cuda.hpp"

namespace distconv
{
namespace tensor
{

template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          typename UnaryFunction>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
TransformReduceSum(Tensor<DataType, Locale, Allocator>& src,
                   Tensor<DataType, Locale, Allocator>& dst,
                   UnaryFunction const& op,
                   h2::gpu::DeviceStream stream = 0)
{
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::ReduceSumFunctor<ND, TensorType, UnaryFunction>()(
    src, src.get_local_shape(), dst, op, stream);
}

template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          typename UnaryFunction>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
TransformReduceSum(Tensor<DataType, Locale, Allocator>& src,
                   Array<ND> const& local_reduction_region,
                   Tensor<DataType, Locale, Allocator>& dst,
                   UnaryFunction const& op,
                   h2::gpu::DeviceStream stream = 0)
{
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::ReduceSumFunctor<ND, TensorType, UnaryFunction>()(
    src, local_reduction_region, dst, op, stream);
}

template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          typename UnaryFunction1,
          typename UnaryFunction2>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
TransformReduceSum(Tensor<DataType, Locale, Allocator>& src,
                   Tensor<DataType, Locale, Allocator>& dst1,
                   UnaryFunction1 const& op1,
                   Tensor<DataType, Locale, Allocator>& dst2,
                   UnaryFunction2 const& op2,
                   h2::gpu::DeviceStream stream = 0)
{
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::
    ReduceSumFunctor2<ND, TensorType, UnaryFunction1, UnaryFunction2>()(
      src, src.get_local_shape(), dst1, op1, dst2, op2, stream);
}

template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          typename UnaryFunction1,
          typename UnaryFunction2>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
TransformReduceSum(Tensor<DataType, Locale, Allocator>& src,
                   Array<ND> const& local_reduction_region,
                   Tensor<DataType, Locale, Allocator>& dst1,
                   UnaryFunction1 const& op1,
                   Tensor<DataType, Locale, Allocator>& dst2,
                   UnaryFunction2 const& op2,
                   h2::gpu::DeviceStream stream = 0)
{
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::
    ReduceSumFunctor2<ND, TensorType, UnaryFunction1, UnaryFunction2>()(
      src, local_reduction_region, dst1, op1, dst2, op2, stream);
}

}  // namespace tensor
}  // namespace distconv
