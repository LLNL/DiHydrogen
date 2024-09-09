#pragma once

#include "distconv/tensor/algorithms/common_cuda.hpp"
#include "distconv/tensor/memory_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

#include <type_traits>
#if __has_include(<nvfunctional>)
#define DISTCONV_HAS_NVFUNCTIONAL_HEADER
#include <nvfunctional>
#endif

namespace distconv
{
namespace tensor
{
namespace algorithms_cuda
{

#ifndef DISTCONV_HAS_NVFUNCTIONAL_HEADER
template <typename F>
class UnaryFunctionWrapper
{
public:
  __host__ __device__ UnaryFunctionWrapper(F func) : m_func{std::move(func)} {}

  template <typename ArgT>
  __host__ __device__ auto operator()(ArgT&& arg) const
  {
    return m_func(std::forward<ArgT>(arg));
  }

  constexpr bool valid() const noexcept { return true; }

private:
  F m_func;
};

// Just don't call it; if you do, it's the identity.
template <>
class UnaryFunctionWrapper<std::nullptr_t>
{
public:
  __host__ __device__ UnaryFunctionWrapper(void*) {}

  template <typename ArgT>
  __host__ __device__ auto operator()(ArgT&& arg) const
  {
    return arg;
  }

  __host__ __device__ constexpr bool valid() const noexcept { return false; }
};
#endif // ifndef DISTCONV_HAS_NVFUNCTIONAL_HEADER

// Generic implementation using atomicAdd
// assumes ND == 3 or 4
template <int ND, typename DataType, typename UnaryFunction, int BLOCK_SIZE>
__global__ static void reduce_kernel(const DataType* src,
                                     Array<ND> src_shape,
                                     Array<ND> src_strides,
                                     DataType* dst,
                                     Array<ND> dst_shape,
                                     Array<ND> dst_strides,
                                     UnaryFunction op,
                                     int thread_work_size)
{
  const int tid = threadIdx.x;
  const int inner_size = src_shape[0] * src_shape[1];
  int inner_idx = tid + blockIdx.x * BLOCK_SIZE * thread_work_size;
  src += blockIdx.y * src_strides[2];
  if (dst_shape[2] != 1)
  {
    dst += blockIdx.y * dst_strides[2];
  }
  if (ND == 4)
  {
    src += blockIdx.z * src_strides[3];
    if (dst_shape[3] != 1)
    {
      dst += blockIdx.z * dst_strides[3];
    }
  }

#ifdef DISTCONV_HAS_NVFUNCTIONAL_HEADER
  nvstd::function<DataType(DataType&)> op_func = op;
  auto const use_op = (op != nullptr);
#else
  UnaryFunctionWrapper<UnaryFunction> op_func(op);
#endif

  for (int i = 0; i < thread_work_size; ++i)
  {
    int idx0 = inner_idx % src_shape[0];
    int idx1 = inner_idx / src_shape[0];
    int tensor_offset = idx0 + idx1 * src_strides[1];
    if (inner_idx < inner_size)
    {
      DataType x = src[tensor_offset];
#ifdef DISTCONV_HAS_NVFUNCTIONAL_HEADER
      if (use_op)
        x = op_func(x);
#else
      if constexpr (op_func.valid())
        x = op_func(x);
#endif
      int dst_idx0 = dst_shape[0] != 1 ? idx0 : 0;
      int dst_idx1 = dst_shape[1] != 1 ? idx1 : 0;
      int dst_offset = dst_idx0 + dst_idx1 * dst_strides[1];
      // printf("dst_offset: %d\n", dst_offset);
      atomicAdd(&dst[dst_offset], x);
    }
    inner_idx += BLOCK_SIZE;
  }
}

inline std::vector<int> find_reduce_dims(const Distribution& src_dist,
                                         const Distribution& dst_dist)
{
  std::vector<int> reduction_dims;
  int nd = src_dist.num_dims();
  for (int i = 0; i < nd; ++i)
  {
    // Reduction is only allowed to a single split
    if (dst_dist.get_split_shape()[i] != 1)
    {
      continue;
    }
    // If there is only one locale, no need to do allreduce
    if (dst_dist.get_locale_shape()[i] == 1)
    {
      continue;
    }
    // If the source already has shared regions, do not reduce
    // again.
    if (src_dist.get_split_shape()[i] != src_dist.get_locale_shape()[i])
    {
      continue;
    }
    reduction_dims.push_back(i);
  }
  return reduction_dims;
}

template <int ND, typename Tensor, typename UnaryFunction>
struct ReduceSumFunctor
{
  int operator()(Tensor& src,
                 const Shape& local_reduction_shape,
                 Tensor& dst,
                 const UnaryFunction& op,
                 h2::gpu::DeviceStream stream)
  {
    using DataType = typename Tensor::data_type;
    if (local_reduction_shape.size() > 0)
    {
      constexpr int block_size = DEFAULT_BLOCK_SIZE;
      constexpr int max_thread_work_size = DEFAULT_MAX_THREAD_WORK_SIZE;
      dim3 block_dims(block_size);
      int thread_work_size = 0;
      dim3 grid_dims(0);
      get_grid_dims<block_size, max_thread_work_size>(
        local_reduction_shape, grid_dims, thread_work_size);

      // const auto src_shape = src.get_local_shape();
      const auto dst_shape = dst.get_local_shape();
      for (int i = 0; i < ND; ++i)
      {
        // reduce to a single entry or no reduction at all
        util::MPIPrintStreamDebug()
          << "dst_shape[" << i << "]: " << dst_shape[i]
          << ", src_shape: " << src.get_local_shape()[i];
        assert_always(dst_shape[i] == 1
                      || dst_shape[i] == src.get_local_shape()[i]);
      }
      const auto src_strides = get_strides<ND>(
        local_reduction_shape, src.get_overlap(), src.get_pitch());
      const auto dst_strides = dst.get_strides();
      reduce_kernel<ND, DataType, UnaryFunction, block_size>
        <<<grid_dims, block_dims, 0, stream>>>(src.get_const_base_ptr(),
                                               Array<ND>(local_reduction_shape),
                                               src_strides,
                                               dst.get_base_ptr(),
                                               dst_shape,
                                               dst_strides,
                                               op,
                                               thread_work_size);
      h2::gpu::sync(stream);
    }

    // Finds the dimensions to reduce. Note that a dimension is not
    // reduced if it includes shared regions, i.e., when split_shape
    // != locale_shape.
    std::vector<int> reduction_dims =
      find_reduce_dims(src.get_distribution(), dst.get_distribution());
    dst.allreduce(reduction_dims);

    return 0;
  }
};

// Generic implementation using atomicAdd
// assumes ND == 3 or 4
template <int ND,
          typename DataType,
          typename UnaryFunction1,
          typename UnaryFunction2,
          int BLOCK_SIZE>
__global__ static void reduce_kernel2(const DataType* src,
                                      Array<ND> src_shape,
                                      Array<ND> src_strides,
                                      DataType* dst1,
                                      Array<ND> dst1_shape,
                                      Array<ND> dst1_strides,
                                      const UnaryFunction1 op1,
                                      DataType* dst2,
                                      Array<ND> dst2_shape,
                                      Array<ND> dst2_strides,
                                      const UnaryFunction2 op2,
                                      int thread_work_size)
{
  const int tid = threadIdx.x;
  const int inner_size = src_shape[0] * src_shape[1];
  int inner_idx = tid + blockIdx.x * BLOCK_SIZE * thread_work_size;
  src += blockIdx.y * src_strides[2];
  if (dst1_shape[2] != 1)
  {
    dst1 += blockIdx.y * dst1_strides[2];
  }
  if (dst2_shape[2] != 1)
  {
    dst2 += blockIdx.y * dst2_strides[2];
  }
  if (ND == 4)
  {
    src += blockIdx.z * src_strides[3];
    if (dst1_shape[3] != 1)
    {
      dst1 += blockIdx.z * dst1_strides[3];
    }
    if (dst2_shape[3] != 1)
    {
      dst2 += blockIdx.z * dst2_strides[3];
    }
  }

#ifdef DISTCONV_HAS_NVFUNCTIONAL_HEADER
  nvstd::function<DataType(DataType)> op1_func = op1;
  nvstd::function<DataType(DataType)> op2_func = op2;
  auto const use_op1 = (op1 != nullptr);
  auto const use_op2 = (op2 != nullptr);
#else
  UnaryFunctionWrapper<UnaryFunction1> op1_func(op1);
  UnaryFunctionWrapper<UnaryFunction2> op2_func(op2);
#endif

  for (int i = 0; i < thread_work_size; ++i)
  {
    int idx0 = inner_idx % src_shape[0];
    int idx1 = inner_idx / src_shape[0];
    int tensor_offset = idx0 + idx1 * src_strides[1];
    if (inner_idx < inner_size)
    {
      const DataType x = src[tensor_offset];
      DataType y1 = x;
#ifdef DISTCONV_HAS_NVFUNCTIONAL_HEADER
      if (use_op1)
      {
        y1 = op1_func(x);
      }
#else
      if constexpr (op1_func.valid())
        y1 = op1_func(x);
#endif
      int dst1_idx0 = dst1_shape[0] != 1 ? idx0 : 0;
      int dst1_idx1 = dst1_shape[1] != 1 ? idx1 : 0;
      int dst1_offset = dst1_idx0 + dst1_idx1 * dst1_strides[1];
      atomicAdd(&dst1[dst1_offset], y1);
      DataType y2 = x;
#ifdef DISTCONV_HAS_NVFUNCTIONAL_HEADER
      if (use_op2)
      {
        y2 = op2_func(x);
      }
#else
      if constexpr (op2_func.valid())
        y2 = op2_func(x);
#endif
      int dst2_idx0 = dst2_shape[0] != 1 ? idx0 : 0;
      int dst2_idx1 = dst2_shape[1] != 1 ? idx1 : 0;
      int dst2_offset = dst2_idx0 + dst2_idx1 * dst2_strides[1];
      atomicAdd(&dst2[dst2_offset], y2);
    }
    inner_idx += BLOCK_SIZE;
  }
}

template <int ND,
          typename Tensor,
          typename UnaryFunction1,
          typename UnaryFunction2>
struct ReduceSumFunctor2
{
  int operator()(Tensor& src,
                 const Shape& local_reduction_shape,
                 Tensor& dst1,
                 const UnaryFunction1& op1,
                 Tensor& dst2,
                 const UnaryFunction2& op2,
                 h2::gpu::DeviceStream stream)
  {
    if (local_reduction_shape.size() > 0)
    {
      constexpr int block_size = DEFAULT_BLOCK_SIZE;
      constexpr int max_thread_work_size = DEFAULT_MAX_THREAD_WORK_SIZE;
      dim3 block_dims(block_size);
      int thread_work_size = 0;
      dim3 grid_dims(0);
      get_grid_dims<block_size, max_thread_work_size>(
        local_reduction_shape, grid_dims, thread_work_size);

      // const auto src_shape = src.get_local_shape();
      const auto dst1_shape = dst1.get_local_shape();
      const auto dst2_shape = dst2.get_local_shape();
      const auto src_strides = get_strides<ND>(
        local_reduction_shape, src.get_overlap(), src.get_pitch());
      const auto dst1_strides = dst1.get_strides();
      const auto dst2_strides = dst2.get_strides();
      reduce_kernel2<ND,
                     typename Tensor::data_type,
                     UnaryFunction1,
                     UnaryFunction2,
                     block_size>
        <<<grid_dims, block_dims, 0, stream>>>(src.get_base_ptr(),
                                               Array<ND>(local_reduction_shape),
                                               src_strides,
                                               dst1.get_base_ptr(),
                                               dst1_shape,
                                               dst1_strides,
                                               op1,
                                               dst2.get_base_ptr(),
                                               dst2_shape,
                                               dst2_strides,
                                               op2,
                                               thread_work_size);
      h2::gpu::sync(stream);
    }

    std::vector<int> reduction_dims =
      find_reduce_dims(src.get_distribution(), dst1.get_distribution());
    dst1.allreduce(reduction_dims);
    reduction_dims =
      find_reduce_dims(src.get_distribution(), dst2.get_distribution());
    dst2.allreduce(reduction_dims);
    return 0;
  }
};

template <int ND, typename DataType, typename Locale, typename Allocator>
void reduction_sanity_check(Tensor<DataType, Locale, Allocator>& src,
                            Tensor<DataType, Locale, Allocator>& dst)
{
  const auto& src_dist = src.get_distribution();
  const auto& dst_dist = dst.get_distribution();
  for (int i = 0; i < ND; ++i)
  {
    assert_always(src_dist.get_locale_shape()[i]
                  == dst_dist.get_locale_shape()[i]);
    if (src.get_shape()[i] == dst.get_shape()[i])
    {
      assert_always(src_dist.get_split_shape()[i]
                    == dst_dist.get_split_shape()[i]);
    }
    else
    {
      assert_always(dst.get_shape()[i] == 1);
      assert_always(dst_dist.get_split_shape()[i] == 1);
    }
  }
}

} // namespace algorithms_cuda

/**
   Each dimension of src must be equal to that of dst or 1, in which
   case src elements are reduced along that dimension. The number of
   splits along the dimension must be 1.
 */
template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          bool CONST>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
ReduceSum(Tensor<DataType, Locale, Allocator>& src,
          Tensor<DataType, Locale, Allocator>& dst,
          h2::gpu::DeviceStream stream = 0)
{
  assert_always(ND == 3 || ND == 4);
  algorithms_cuda::reduction_sanity_check(src, dst);
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::ReduceSumFunctor<ND, TensorType, std::nullptr_t>()(
    src, src.get_local_shape(), dst, nullptr, stream);
}

// Kind of an ugly hack to allow only partial reduction per local subregion
template <int ND, typename DataType, typename Locale, typename Allocator>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
ReduceSum(Tensor<DataType, Locale, Allocator>& src,
          const Shape& local_reduction_region,
          Tensor<DataType, Locale, Allocator>& dst,
          h2::gpu::DeviceStream stream = 0)
{
  assert_always(ND == 3 || ND == 4);
  algorithms_cuda::reduction_sanity_check<ND, DataType, Locale, Allocator>(src,
                                                                           dst);
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::ReduceSumFunctor<ND, TensorType, std::nullptr_t>()(
    src, local_reduction_region, dst, nullptr, stream);
}

template <int ND, typename DataType, typename Locale, typename Allocator>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
ReduceSum(Tensor<DataType, Locale, Allocator>& src,
          Tensor<DataType, Locale, Allocator>& dst1,
          Tensor<DataType, Locale, Allocator>& dst2,
          h2::gpu::DeviceStream stream = 0)
{
  assert_always(ND == 3 || ND == 4);
  algorithms_cuda::reduction_sanity_check(src, dst1);
  algorithms_cuda::reduction_sanity_check(src, dst2);
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::
    ReduceSumFunctor2<ND, TensorType, std::nullptr_t, std::nullptr_t>()(
      src, src.get_local_shape(), dst1, nullptr, dst2, nullptr, stream);
}

template <int ND,
          typename DataType,
          typename Locale,
          typename Allocator,
          bool CONST>
typename std::enable_if<std::is_same<Allocator, CUDAAllocator>::value,
                        int>::type
ReduceSum(Tensor<DataType, Locale, Allocator>& src,
          const Array<ND>& local_reduction_region,
          Tensor<DataType, Locale, Allocator>& dst1,
          Tensor<DataType, Locale, Allocator>& dst2,
          h2::gpu::DeviceStream stream = 0)
{
  assert_always(ND == 3 || ND == 4);
  algorithms_cuda::reduction_sanity_check(src, dst1);
  algorithms_cuda::reduction_sanity_check(src, dst2);
  using TensorType = Tensor<DataType, Locale, Allocator>;
  return algorithms_cuda::
    ReduceSumFunctor2<ND, TensorType, std::nullptr_t, std::nullptr_t>()(
      src, local_reduction_region, dst1, nullptr, dst2, nullptr, stream);
}

} // namespace tensor
} // namespace distconv
