#pragma once

#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/algorithms/common_cuda.hpp"

#include <type_traits>

namespace distconv {
namespace tensor {
namespace algorithms_cuda {

template <int ND, typename DataType, int BLOCK_SIZE,
          int INNER_DIM,
          typename TransformFunc>
__global__ void transform_kernel(
    Array<ND> shape,
    Array<ND> strides,
    DataType *data,
    TransformFunc op,
    int thread_work_size,
    int num_inner_blocks) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  int inner_bid = bid % num_inner_blocks;
  bid /= num_inner_blocks;
  int inner_idx = tid + inner_bid * BLOCK_SIZE * thread_work_size;
  int inner_size = 1;
#pragma unroll
  for (int i = 0; i <= INNER_DIM; ++i) {
    inner_size *= shape[i];
  }
#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    data += strides[i] * (bid % shape[i]);
    bid /= shape[i];
  }

  for (int i = 0; i < thread_work_size; ++i) {
    int tensor_offset = 0;
    int inner_idx_i = inner_idx;
#pragma unroll
    for (int j = 0; j <= INNER_DIM; ++j) {
      int idx_j = inner_idx_i % shape[j];
      tensor_offset += strides[j] * idx_j;
      inner_idx_i /= shape[j];
    }
    if (inner_idx < inner_size) {
      op(data[tensor_offset]);
    }
    inner_idx += BLOCK_SIZE;
  }
}

template <typename DataType, int BLOCK_SIZE, int INNER_DIM,
          typename TransformFunc>
void transform(Shape shape, IndexVector strides, DataType *data,
               TransformFunc op, int thread_work_size, int num_inner_blocks,
               const dim3 &grid_dims, const dim3 &block_dims,
               cudaStream_t stream) {
  const int nd = shape.num_dims();

#define CALL_KERNEL(ND)                                                 \
  transform_kernel<ND, DataType, BLOCK_SIZE, INNER_DIM, TransformFunc>  \
      <<<grid_dims, block_dims, 0, stream>>>(                           \
          Array<ND>(shape), Array<ND>(strides),                         \
          data, op, thread_work_size, num_inner_blocks)

  switch (nd) {
    case 3:
      CALL_KERNEL(3);
      break;
    case 4:
      CALL_KERNEL(4);
      break;
    case 5:
      CALL_KERNEL(5);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
}

template <int ND, typename DataType1, typename DataType2,
          int BLOCK_SIZE, int INNER_DIM,
          typename TransformFunc>
__global__ void transform_kernel(Array<ND> shape, Array<ND> strides,
                                 DataType1 *data1, DataType2 *data2,
                                 TransformFunc op,
                                 int thread_work_size, int num_inner_blocks) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  int inner_bid = bid % num_inner_blocks;
  bid /= num_inner_blocks;
  int inner_idx = tid + inner_bid * BLOCK_SIZE * thread_work_size;
  int inner_size = 1;
#pragma unroll
  for (int i = 0; i <= INNER_DIM; ++i) {
    inner_size *= shape[i];
  }
#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    index_t offset = strides[i] * (bid % shape[i]);
    data1 += offset;
    data2 += offset;
    bid /= shape[i];
  }

  for (int i = 0; i < thread_work_size; ++i) {
    int tensor_offset = 0;
    int inner_idx_i = inner_idx;
#pragma unroll
    for (int j = 0; j <= INNER_DIM; ++j) {
      int idx_j = inner_idx_i % shape[j];
      tensor_offset += strides[j] * idx_j;
      inner_idx_i /= shape[j];
    }
    if (inner_idx < inner_size) {
      op(data1[tensor_offset],
         data2[tensor_offset]);
    }
    inner_idx += BLOCK_SIZE;
  }
}

template <typename DataType1, typename DataType2, int BLOCK_SIZE, int INNER_DIM,
          typename TransformFunc>
void transform(Shape shape, IndexVector strides,
               DataType1 *data1, DataType2 *data2,
               TransformFunc op,
               int thread_work_size, int num_inner_blocks,
               const dim3 &grid_dims, const dim3 &block_dims,
               cudaStream_t stream) {
  const int nd = shape.num_dims();

#define CALL_KERNEL(ND)                                                 \
  transform_kernel<ND, DataType1, DataType2, BLOCK_SIZE, INNER_DIM,     \
                   TransformFunc>                                       \
      <<<grid_dims, block_dims, 0, stream>>>(                           \
          Array<ND>(shape), Array<ND>(strides),                         \
          data1, data2,                                                 \
          op, thread_work_size, num_inner_blocks)

  switch (nd) {
    case 3:
      CALL_KERNEL(3);
      break;
    case 4:
      CALL_KERNEL(4);
      break;
    case 5:
      CALL_KERNEL(5);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
}

template <int ND, typename DataType1, typename DataType2, typename DataType3,
          int BLOCK_SIZE, int INNER_DIM, typename TransformFunc>
__global__ void transform_kernel(Array<ND> shape, Array<ND> strides,
                                 DataType1 *data1, DataType2 *data2,
                                 DataType3 *data3,
                                 TransformFunc op,
                                 int thread_work_size, int num_inner_blocks) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  int inner_bid = bid % num_inner_blocks;
  bid /= num_inner_blocks;
  int inner_idx = tid + inner_bid * BLOCK_SIZE * thread_work_size;
  int inner_size = 1;
#pragma unroll
  for (int i = 0; i <= INNER_DIM; ++i) {
    inner_size *= shape[i];
  }
#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    index_t offset = strides[i] * (bid % shape[i]);
    data1 += offset;
    data2 += offset;
    data3 += offset;
    bid /= shape[i];
  }

  for (int i = 0; i < thread_work_size; ++i) {
    int tensor_offset = 0;
    int inner_idx_i = inner_idx;
#pragma unroll
    for (int j = 0; j <= INNER_DIM; ++j) {
      int idx_j = inner_idx_i % shape[j];
      tensor_offset += strides[j] * idx_j;
      inner_idx_i /= shape[j];
    }
    if (inner_idx < inner_size) {
      op(data1[tensor_offset],
         data2[tensor_offset],
         data3[tensor_offset]);
    }
    inner_idx += BLOCK_SIZE;
  }
}

template <typename DataType1, typename DataType2, typename DataType3,
          int BLOCK_SIZE, int INNER_DIM, typename TransformFunc>
void transform(Shape shape, IndexVector strides,
               DataType1 *data1, DataType2 *data2,
               DataType3 *data3,
               TransformFunc op,
               int thread_work_size, int num_inner_blocks,
               const dim3 &grid_dims, const dim3 &block_dims,
               cudaStream_t stream) {
  const int nd = shape.num_dims();

#define CALL_KERNEL(ND)                                                 \
  transform_kernel<ND, DataType1, DataType2, DataType3, BLOCK_SIZE,     \
                   INNER_DIM, TransformFunc>                            \
      <<<grid_dims, block_dims, 0, stream>>>(                           \
          Array<ND>(shape), Array<ND>(strides),                         \
          data1, data2, data3,                                          \
          op, thread_work_size, num_inner_blocks)

  switch (nd) {
    case 3:
      CALL_KERNEL(3);
      break;
    case 4:
      CALL_KERNEL(4);
      break;
    case 5:
      CALL_KERNEL(5);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
}

template <int ND, typename DataType1, typename DataType2, typename DataType3,
          typename DataType4, int BLOCK_SIZE, int INNER_DIM,
          typename TransformFunc>
__global__ void transform_kernel(Array<ND> shape, Array<ND> strides,
                                 DataType1 *data1, DataType2 *data2,
                                 DataType3 *data3, DataType4 *data4,
                                 TransformFunc op,
                                 int thread_work_size, int num_inner_blocks) {
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  int inner_bid = bid % num_inner_blocks;
  bid /= num_inner_blocks;
  int inner_idx = tid + inner_bid * BLOCK_SIZE * thread_work_size;
  int inner_size = 1;
#pragma unroll
  for (int i = 0; i <= INNER_DIM; ++i) {
    inner_size *= shape[i];
  }
#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    index_t offset = strides[i] * (bid % shape[i]);
    data1 += offset;
    data2 += offset;
    data3 += offset;
    data4 += offset;
    bid /= shape[i];
  }

  for (int i = 0; i < thread_work_size; ++i) {
    int tensor_offset = 0;
    int inner_idx_i = inner_idx;
#pragma unroll
    for (int j = 0; j <= INNER_DIM; ++j) {
      int idx_j = inner_idx_i % shape[j];
      tensor_offset += strides[j] * idx_j;
      inner_idx_i /= shape[j];
    }
    if (inner_idx < inner_size) {
      op(data1[tensor_offset],
         data2[tensor_offset],
         data3[tensor_offset],
         data4[tensor_offset]);
    }
    inner_idx += BLOCK_SIZE;
  }
}

template <typename DataType1, typename DataType2, typename DataType3,
          typename DataType4, int BLOCK_SIZE, int INNER_DIM,
          typename TransformFunc>
void transform(Shape shape, IndexVector strides,
               DataType1 *data1, DataType2 *data2,
               DataType3 *data3, DataType4 *data4,
               TransformFunc op,
               int thread_work_size, int num_inner_blocks,
               const dim3 &grid_dims, const dim3 &block_dims,
               cudaStream_t stream) {
  const int nd = shape.num_dims();
#define CALL_KERNEL(ND)                                                 \
  transform_kernel<ND, DataType1, DataType2, DataType3, DataType4,      \
                   BLOCK_SIZE, INNER_DIM, TransformFunc>                \
      <<<grid_dims, block_dims, 0, stream>>>(                           \
          Array<ND>(shape), Array<ND>(strides),                         \
          data1, data2, data3, data4,                                   \
          op, thread_work_size, num_inner_blocks)
  switch (nd) {
    case 3:
      CALL_KERNEL(3);
      break;
    case 4:
      CALL_KERNEL(4);
      break;
    case 5:
      CALL_KERNEL(5);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
}

} // namespace algorithms_cuda

template <typename Tensor, typename TransformFunc>
typename std::enable_if<
  std::is_same<typename Tensor::allocator_type,
               CUDAAllocator>::value,
  int>::type
Transform(Tensor &tensor, TransformFunc op,
          cudaStream_t stream=0) {
  namespace algo = algorithms_cuda;
  if (tensor.get_local_size() == 0) return 0;

  constexpr int block_size= algo::DEFAULT_BLOCK_SIZE;
  constexpr int max_thread_work_size=
      algo::DEFAULT_MAX_THREAD_WORK_SIZE;
  dim3 block_dims(block_size);
  int thread_work_size;
  dim3 grid_dims(0);
  int inner_dim;
  int num_inner_blocks;
  algo::get_grid_dims2<block_size, max_thread_work_size>(
      tensor.get_local_shape(), grid_dims, thread_work_size,
      inner_dim, num_inner_blocks);

  const auto shape = tensor.get_local_shape();
  const auto strides = get_strides(shape,
                                   tensor.get_overlap(),
                                   tensor.get_pitch());

  util::MPIPrintStreamDebug() << "grid_dim: " << grid_dims.x
                              << ", inner dim: " << inner_dim
                              << ", num_inner_blocks: " << num_inner_blocks;

  if (tensor.get_num_dims() > 5) {
    // The below switch block assumes ND <= 5. Otherwise, inner_dim
    // can be >= 5, and the default case would hit. Simply repeating
    // the case block would work.
    util::MPIPrintStreamError() <<
        "Tensors with 6 or larger number of dimensions not supported.";
    throw std::exception();
  }
  using DataType = typename Tensor::data_type;

#define CALL_TRANFORM(INNER_DIM)                                        \
  algo::transform<DataType, block_size, INNER_DIM, TransformFunc>(      \
      shape, strides, tensor.get_base_ptr(), op,                        \
      thread_work_size, num_inner_blocks, grid_dims, block_dims, stream)

  switch (inner_dim) {
    case 0:
      CALL_TRANFORM(0);
      break;
    case 1:
      CALL_TRANFORM(1);
      break;
    case 2:
      CALL_TRANFORM(2);
      break;
    case 3:
      CALL_TRANFORM(3);
      break;
    case 4:
      CALL_TRANFORM(4);
      break;
    default:
      throw std::exception();
  }
#undef CALL_TRANFORM
  return 0;
}

template <typename Tensor1, typename Tensor2, typename TransformFunc>
typename std::enable_if<
  std::is_same<typename std::remove_const<Tensor1>::type::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename std::remove_const<Tensor2>::type::allocator_type,
               CUDAAllocator>::value,
  int>::type
Transform(Tensor1 &tensor1, Tensor2 &tensor2,
          TransformFunc op, cudaStream_t stream=0) {
  namespace algo = algorithms_cuda;

  // All tensor arguments have an eaual shape
  assert_eq(tensor1.get_local_shape(), tensor2.get_local_shape());
  // All tensor arguments have the same distribution
  assert_eq(tensor1.get_distribution(), tensor2.get_distribution());

  if (tensor1.get_local_size() == 0) return 0;

  constexpr int block_size= algo::DEFAULT_BLOCK_SIZE;
  constexpr int max_thread_work_size=
      algo::DEFAULT_MAX_THREAD_WORK_SIZE;
  dim3 block_dims(block_size);
  int thread_work_size;
  dim3 grid_dims(0);
  int inner_dim;
  int num_inner_blocks;
  algo::get_grid_dims2<block_size, max_thread_work_size>(
      tensor1.get_local_shape(), grid_dims, thread_work_size,
      inner_dim, num_inner_blocks);

  const auto shape = tensor1.get_local_shape();
  const auto strides = get_strides(shape,
                                   tensor1.get_overlap(),
                                   tensor1.get_pitch());

  util::MPIPrintStreamDebug() << "grid_dim: " << grid_dims.x
                              << ", inner dim: " << inner_dim
                              << ", num_inner_blocks: " << num_inner_blocks
                              << ", shape: " << shape
                              << ", strides: " << strides;

  if (tensor1.get_num_dims() > 5) {
    // The below switch block assumes ND <= 5. Otherwise, inner_dim
    // can be >= 5, and the default case would hit. Simply repeating
    // the case block would work.
    util::MPIPrintStreamError() <<
        "Tensors with 6 or larger number of dimensions not supported.";
    throw std::exception();
  }
  using DataType1 = typename std::conditional<
    std::is_const<Tensor1>::value,
    typename std::add_const<typename std::remove_const<Tensor1>::type::data_type>::type,
    typename Tensor1::data_type>::type;
  using DataType2 = typename std::conditional<
    std::is_const<Tensor2>::value,
    typename std::add_const<typename std::remove_const<Tensor2>::type::data_type>::type,
    typename Tensor2::data_type>::type;

#define CALL_TRANFORM(INNER_DIM) \
  algo::transform<DataType1, DataType2, block_size, INNER_DIM, TransformFunc>( \
      shape, strides, tensor1.get_base_ptr(), tensor2.get_base_ptr(),   \
      op, thread_work_size, num_inner_blocks, grid_dims, block_dims, stream);
  switch (inner_dim) {
    case 0:
      CALL_TRANFORM(0);
      break;
    case 1:
      CALL_TRANFORM(1);
      break;
    case 2:
      CALL_TRANFORM(2);
      break;
    case 3:
      CALL_TRANFORM(3);
      break;
    case 4:
      CALL_TRANFORM(4);
      break;
    default:
      throw std::exception();
  }
#undef CALL_TRANFORM
  return 0;
}

template <typename Tensor1, typename Tensor2, typename Tensor3,
          typename TransformFunc>
typename std::enable_if<
  std::is_same<typename Tensor1::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename Tensor2::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename Tensor3::allocator_type,
               CUDAAllocator>::value,
  int>::type
Transform(Tensor1 &tensor1, Tensor2 &tensor2, Tensor3 &tensor3,
          TransformFunc op, cudaStream_t stream=0) {
  namespace algo = algorithms_cuda;

  // All tensor arguments have an eaual shape
  assert_eq(tensor1.get_local_shape(), tensor2.get_local_shape());
  assert_eq(tensor2.get_local_shape(), tensor3.get_local_shape());
  assert_eq(tensor3.get_local_shape(), tensor1.get_local_shape());
  // All tensor arguments have the same distribution
  assert_eq(tensor1.get_distribution(), tensor2.get_distribution());
  assert_eq(tensor2.get_distribution(), tensor3.get_distribution());
  assert_eq(tensor3.get_distribution(), tensor1.get_distribution());

  if (tensor1.get_local_size() == 0) return 0;

  constexpr int block_size= algo::DEFAULT_BLOCK_SIZE;
  constexpr int max_thread_work_size=
      algo::DEFAULT_MAX_THREAD_WORK_SIZE;
  dim3 block_dims(block_size);
  int thread_work_size;
  dim3 grid_dims(0);
  int inner_dim;
  int num_inner_blocks;
  algo::get_grid_dims2<block_size, max_thread_work_size>(
      tensor1.get_local_shape(), grid_dims, thread_work_size,
      inner_dim, num_inner_blocks);

  const auto shape = tensor1.get_local_shape();
  const auto strides = get_strides(shape,
                                   tensor1.get_overlap(),
                                   tensor1.get_pitch());

  util::MPIPrintStreamDebug() << "grid_dim: " << grid_dims.x
                              << ", inner dim: " << inner_dim
                              << ", num_inner_blocks: " << num_inner_blocks;

  using DataType1 = typename Tensor1::data_type;
  using DataType2 = typename Tensor2::data_type;
  using DataType3 = typename Tensor3::data_type;
  if (tensor1.get_num_dims() > 5) {
    // The below switch block assumes ND <= 5. Otherwise, inner_dim
    // can be >= 5, and the default case would hit. Simply repeating
    // the case block would work.
    util::MPIPrintStreamError() <<
        "Tensors with 6 or larger number of dimensions not supported.";
    throw std::exception();
  }
#define CALL_TRANFORM(INNER_DIM)                                        \
  algo::transform<DataType1, DataType2, DataType3, block_size,          \
                  INNER_DIM, TransformFunc>(                            \
      shape, strides, tensor1.get_base_ptr(), tensor2.get_base_ptr(),   \
      tensor3.get_base_ptr(), op, thread_work_size, num_inner_blocks,   \
      grid_dims, block_dims, stream)

  switch (inner_dim) {
    case 0:
      CALL_TRANFORM(0);
      break;
    case 1:
      CALL_TRANFORM(1);
      break;
    case 2:
      CALL_TRANFORM(2);
      break;
    case 3:
      CALL_TRANFORM(3);
      break;
    case 4:
      CALL_TRANFORM(4);
      break;
    default:
      throw std::exception();
  }
#undef CALL_TRANFORM
  return 0;
}

template <typename Tensor1, typename Tensor2, typename Tensor3,
          typename Tensor4, typename TransformFunc>
typename std::enable_if<
  std::is_same<typename Tensor1::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename Tensor2::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename Tensor3::allocator_type,
               CUDAAllocator>::value &&
  std::is_same<typename Tensor4::allocator_type,
               CUDAAllocator>::value,
  int>::type
Transform(Tensor1 &tensor1, Tensor2 &tensor2, Tensor3 &tensor3,
          Tensor4 &tensor4, TransformFunc op,
          cudaStream_t stream=0) {
  namespace algo = algorithms_cuda;

  // All tensor arguments have an eaual shape
  assert_always(tensor1.get_local_shape() == tensor2.get_local_shape());
  assert_always(tensor2.get_local_shape() == tensor3.get_local_shape());
  assert_always(tensor3.get_local_shape() == tensor4.get_local_shape());
  assert_always(tensor4.get_local_shape() == tensor1.get_local_shape());
  // All tensor arguments have the same distribution
  assert_eq(tensor1.get_distribution(), tensor2.get_distribution());
  assert_eq(tensor2.get_distribution(), tensor3.get_distribution());
  assert_eq(tensor3.get_distribution(), tensor4.get_distribution());
  assert_eq(tensor4.get_distribution(), tensor1.get_distribution());

  if (tensor1.get_local_size() == 0) return 0;

  constexpr int block_size= algo::DEFAULT_BLOCK_SIZE;
  constexpr int max_thread_work_size=
      algo::DEFAULT_MAX_THREAD_WORK_SIZE;
  dim3 block_dims(block_size);
  int thread_work_size;
  dim3 grid_dims(0);
  int inner_dim;
  int num_inner_blocks;
  algo::get_grid_dims2<block_size, max_thread_work_size>(
      tensor1.get_local_shape(), grid_dims, thread_work_size,
      inner_dim, num_inner_blocks);

  const auto shape = tensor1.get_local_shape();
  const auto strides = get_strides(shape,
                                   tensor1.get_overlap(),
                                   tensor1.get_pitch());

  util::MPIPrintStreamDebug() << "grid_dim: " << grid_dims.x
                              << ", inner dim: " << inner_dim
                              << ", num_inner_blocks: " << num_inner_blocks
                              << "\n";

  if (tensor1.get_num_dims() > 5) {
    // The below switch block assumes ND <= 5. Otherwise, inner_dim
    // can be >= 5, and the default case would hit. Simply repeating
    // the case block would work.
    util::MPIPrintStreamError() <<
        "Tensors with 6 or larger number of dimensions not supported.";
    throw std::exception();
  }
  using DataType1 = typename Tensor1::data_type;
  using DataType2 = typename Tensor2::data_type;
  using DataType3 = typename Tensor3::data_type;
  using DataType4 = typename Tensor4::data_type;
#define CALL_TRANFORM(INNER_DIM)                                        \
  algo::transform<DataType1, DataType2, DataType3, DataType4,           \
                  block_size, INNER_DIM, TransformFunc>(                \
      shape, strides, tensor1.get_base_ptr(), tensor2.get_base_ptr(),   \
      tensor3.get_base_ptr(), tensor4.get_base_ptr(),                   \
      op, thread_work_size, num_inner_blocks, grid_dims, block_dims, stream);

  switch (inner_dim) {
    case 0:
      CALL_TRANFORM(0);
      break;
    case 1:
      CALL_TRANFORM(1);
      break;
    case 2:
      CALL_TRANFORM(2);
      break;
    case 3:
      CALL_TRANFORM(3);
      break;
    case 4:
      CALL_TRANFORM(4);
      break;
    default:
      throw std::exception();
  }
#undef CALL_TRANFORM
  return 0;
}

} // namespace tensor
} // namespace distconv
