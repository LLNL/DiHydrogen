#pragma once

#include <distconv_config.hpp>

#include "distconv/base.hpp"
#include "distconv/runtime_gpu.hpp"

#if H2_HAS_CUDA
#include <cooperative_groups.h>
#define GPU_LAST_ERROR cudaGetLastError
#define GPU_LAUNCH_COOP_KERNEL cudaLaunchCooperativeKernel
#define GPU_SUCCESS cudaSuccess
#elif H2_HAS_ROCM
#include <hip/hip_cooperative_groups.h>
#define GPU_LAST_ERROR hipGetLastError
#define GPU_LAUNCH_COOP_KERNEL hipLaunchCooperativeKernel
#define GPU_SUCCESS hipSuccess
#endif

namespace distconv {
namespace tensor {

enum class HaloTraversalOpGroup {THREAD, WARP, BLOCK};

namespace internal {

template <int ND, typename DataType, typename OpType>
__global__
typename std::enable_if<OpType::group == HaloTraversalOpGroup::THREAD, void>::type
traverse_halo_generic_kernel(DataType *tensor,
                             Array<ND> shape,
                             int dim, Side side,
                             bool inner, int halo_width,
                             size_t num_halo_points,
                             OpType op) {
  const size_t num_threads = blockDim.x * gridDim.x;
  const bool fwd_halo = side == Side::RHS;
  for (size_t packed_offset = threadIdx.x +
           blockIdx.x * blockDim.x;
       packed_offset < num_halo_points; packed_offset +=num_threads) {
    size_t tensor_offset = 0;
    size_t dim_offset = 1;
    size_t offset = packed_offset;
#pragma unroll
    for (int i = 0; i < ND; ++i) {
      int i_dim = i == dim ? halo_width : shape[i];
      int idx = offset % i_dim;
      if (i == dim) {
        if (fwd_halo) {
          if (inner) {
            idx += shape[i] - halo_width * 2;
          } else {
            idx += shape[i] - halo_width;
          }
        } else {
          if (inner) {
            idx += halo_width;
          }
        }
      }
      tensor_offset += idx * dim_offset;
      offset /= i_dim;
      dim_offset *= shape[i];
    }
    op(tensor[tensor_offset], packed_offset);
  }
}

template <int ND, typename DataType, typename OpType>
__global__
typename std::enable_if<OpType::group == HaloTraversalOpGroup::BLOCK, void>::type
traverse_halo_generic_kernel(DataType *tensor,
                             Array<ND> shape,
                             int dim, Side side,
                             bool inner, int halo_width,
                             size_t num_halo_points,
                             OpType op) {
  // TODO: This should not be called as it is not yet supported.
}

template <typename DataType, typename OpType>
void traverse_halo_generic(DataType *tensor, const Shape &shape,
                           int dim, Side side, bool inner, int halo_width,
                           size_t num_halo_points, OpType op, const int nd,
                           const dim3 &grid_dims, const dim3 &block_dims,
                           h2::gpu::DeviceStream s) {
#define CALL_KERNEL(ND) \
  traverse_halo_generic_kernel<ND, DataType, OpType>                    \
      <<<grid_dims, block_dims, 0, s>>>(                                \
          tensor, Array<ND>(shape), dim, side, inner, halo_width,       \
          num_halo_points, op)

  switch (nd) {
    case 1:
      CALL_KERNEL(1);
      break;
    case 2:
      CALL_KERNEL(2);
      break;
    case 3:
      CALL_KERNEL(3);
      break;
    case 4:
      CALL_KERNEL(4);
      break;
    case 5:
      CALL_KERNEL(5);
      break;
    case 6:
      CALL_KERNEL(6);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
}

// ND: 4, 5
// Traverse halo at dimension 0
template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::THREAD, void>::type
traverse_halo_opt_dim0_apply(DataType *tensor, size_t tensor_offset,
                             size_t packed_offset, int x_len, int y_len,
                             int halo_width, OpType op) {
  for (int y = threadIdx.y; y < y_len; y += blockDim.y) {
    op(tensor[tensor_offset+y*x_len+threadIdx.x],
       packed_offset + y * halo_width + threadIdx.x);
  }
}

template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::BLOCK, void>::type
traverse_halo_opt_dim0_apply(DataType *tensor, size_t tensor_offset,
                             size_t packed_offset, int x_len, int y_len,
                             int halo_width, OpType op) {
  for (int y = 0; y < y_len; y += blockDim.y) {
    auto tensor_idx = min(y + threadIdx.y, y_len - 1);
    op(tensor[tensor_offset+tensor_idx*x_len+threadIdx.x],
       packed_offset + y * halo_width, threadIdx.y * blockDim.x + threadIdx.x,
       min(blockDim.y, y_len - y) * blockDim.x);
  }
}

template <int ND, typename DataType, typename OpType, bool inner, Side side>
__device__ static
void traverse_halo_opt_dim0(DataType *tensor, Array<ND> shape,
                            int halo_width, OpType op) {
  constexpr int dim = 0;
  auto sample_idx = blockIdx.y;
  auto ch_offset = ND == 5 ? shape[1] * shape[2]: shape[1];
  auto sample_offset = ch_offset * shape[-2];
  int packed_halo_idx = threadIdx.x;
  int halo_idx = threadIdx.x;
  if (side == Side::RHS) {
    if (inner) {
      halo_idx += (int)(shape[dim]) - halo_width * 2;
    } else {
      halo_idx += (int)(shape[dim]) - halo_width;
    }
  } else {
    if (inner) {
      halo_idx += halo_width;
    }
  }
  size_t offset_common = sample_idx * sample_offset + blockIdx.x * ch_offset;
  ch_offset *= gridDim.x;
  for (int ch_idx = blockIdx.x; ch_idx < shape[-2]; ch_idx += gridDim.x) {
    if (ND == 5) {
      for (int z = 0; z < shape[2]; ++z) {
        for (int y = threadIdx.y; y < shape[1]; y += blockDim.y) {
          // TODO
#if 0
          size_t packed_offset = (offset_common + y) * halo_width + packed_halo_idx;
          size_t tensor_offset = (offset_common + y) * shape[0] + halo_idx;
          op(tensor[tensor_offset], packed_offset);
#endif
        }
        offset_common += shape[1];
      }
    } else {
      size_t packed_offset = offset_common * halo_width + packed_halo_idx
          - threadIdx.x;
      size_t tensor_offset = offset_common * shape[0] + halo_idx
          - threadIdx.x;
      traverse_halo_opt_dim0_apply(tensor, tensor_offset, packed_offset,
                                   shape[0], shape[1], halo_width, op);
    }
    offset_common += ch_offset;
  }
}

// ND: 4, 5
// Traverse halo at dimension 1 with a known halo width
template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::THREAD, void>::type
traverse_halo_opt_dim1_apply(DataType * __restrict__ tensor,
                             size_t tensor_offset,
                             size_t packed_offset,
                             int len, OpType op) {
  for (int x = threadIdx.x; x < len; x += blockDim.x) {
    op(tensor[tensor_offset+x], packed_offset + x);
  }
}

template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::BLOCK, void>::type
traverse_halo_opt_dim1_apply(DataType * __restrict__ tensor,
                             size_t tensor_offset,
                             size_t packed_offset,
                             int len, OpType op) {
  for (int x = 0; x < len; x += blockDim.x) {
    // avoid reading beyond array boundary
    auto tensor_idx = min(x + threadIdx.x, len - 1);
    op(tensor[tensor_offset+tensor_idx], packed_offset + x,
       threadIdx.x, min(blockDim.x, len - x));
  }
}

template <typename DataType, typename OpType, int halo_width>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::THREAD, void>::type
traverse_halo_opt_dim1_5d_apply(DataType * __restrict__ tensor,
                                size_t tensor_offset_common,
                                size_t packed_offset_common,
                                int x_len, int y_len, int z_len,
                                int halo_tensor_base,
                                OpType op) {
  packed_offset_common += x_len * threadIdx.y;
  tensor_offset_common += x_len * y_len * threadIdx.y;
  for (int y = threadIdx.y; y < z_len; y += blockDim.y) {
    int const z_block_len = min(static_cast<int>(blockDim.y), z_len - (y - static_cast<int>(threadIdx.y)));
#pragma unroll
    for (int hw = 0; hw < halo_width; ++hw) {
      size_t packed_offset = packed_offset_common + x_len * z_block_len * hw;
      int halo_idx = hw + halo_tensor_base;
      size_t tensor_offset = tensor_offset_common + x_len * halo_idx;
      for (int x = threadIdx.x; x < x_len; x += blockDim.x) {
        op(tensor[tensor_offset+x], packed_offset + x);
      }
    }
    packed_offset_common += halo_width * x_len * blockDim.y;
    tensor_offset_common += x_len * y_len * blockDim.y;
  }
}

template <typename DataType, typename OpType, int halo_width>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::BLOCK, void>::type
traverse_halo_opt_dim1_5d_apply(DataType * __restrict__ tensor,
                                size_t tensor_offset_common,
                                size_t packed_offset_common,
                                int x_len, int y_len, int z_len,
                                int halo_tensor_base,
                                OpType op) {
  // Note that the traversal below assumes that the X dimension is
  // completely covered by a single thread block, i.e., blockDim.x == x_len.
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  for (int z = 0; z < z_len; z += blockDim.y) {
    int z_thread = min(z + threadIdx.y, z_len - 1);
    int z_block_len = min(blockDim.y, z_len - z);
#pragma unroll
    for (int hw = 0; hw < halo_width; ++hw) {
      size_t packed_offset = packed_offset_common + x_len * z_block_len * hw
          + z * x_len * halo_width;
      int halo_idx = hw + halo_tensor_base;
      size_t tensor_offset = tensor_offset_common + x_len * halo_idx
          + z_thread * x_len * y_len;
      for (int x = 0; x < x_len; x += blockDim.x) {
        // avoid reading beyond array boundary
        auto x_thread = min(x + threadIdx.x, x_len - 1);
        op(tensor[tensor_offset+x_thread], packed_offset + x,
           tid, min(blockDim.x, x_len - x) * z_block_len);
      }
    }
  }
}

template <int ND, typename DataType, typename OpType,
          bool inner, Side side, int halo_width>
__device__ static void
traverse_halo_opt_dim1(DataType * __restrict__ tensor, const Array<ND> &shape, OpType op) {
  constexpr int dim = 1;
  auto sample_idx = blockIdx.y;
  auto ch_offset = ND == 5 ? shape[0] * shape[2]: shape[0];
  const auto sample_offset = ch_offset * shape[-2];
  int halo_tensor_base = 0;
  if (side == Side::RHS) {
    if (inner) {
      halo_tensor_base += (int)(shape[dim]) - halo_width * 2;
    } else {
      halo_tensor_base += (int)(shape[dim]) - halo_width;
    }
  } else {
    if (inner) {
      halo_tensor_base += halo_width;
    }
  }
  size_t offset_common = sample_idx * sample_offset + blockIdx.x * ch_offset;
  // The loop below traverses the channel dimension by gridDim.x
  // blocks, so the channel offset needs to be multiplied by the
  // number of blocks.
  ch_offset *= gridDim.x;
  for (int ch_idx = blockIdx.x; ch_idx < shape[-2]; ch_idx += gridDim.x) {
    const size_t packed_offset_common = offset_common * halo_width;
    const size_t tensor_offset_common = offset_common * shape[1];
    if (ND == 5) {
      traverse_halo_opt_dim1_5d_apply<DataType, OpType, halo_width>(
          tensor, tensor_offset_common,
          packed_offset_common,
          shape[0], shape[1], shape[2],
          halo_tensor_base, op);
    } else {
#pragma unroll
      for (int hw = 0; hw < halo_width; ++hw) {
        int halo_idx = hw + halo_tensor_base;
        size_t packed_offset = packed_offset_common + shape[0] * hw;
        size_t tensor_offset = tensor_offset_common + shape[0] * halo_idx;
        traverse_halo_opt_dim1_apply(tensor, tensor_offset, packed_offset,
                                     shape[0], op);
      }
    }
    offset_common += ch_offset;
  }
}

// ND: 5
// Traverse halo at dimension 1 with a known halo width
template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::THREAD, void>::type
traverse_halo_opt_dim2_apply(DataType * __restrict__ tensor,
                             size_t tensor_offset_common,
                             size_t packed_offset_common,
                             int x_len, int y_len, OpType op) {
  for (int y = threadIdx.y; y < y_len; y += blockDim.y) {
    size_t packed_offset = packed_offset_common + x_len * y;
    size_t tensor_offset = tensor_offset_common + x_len * y;
    for (int x = threadIdx.x; x < x_len; x += blockDim.x) {
      op(tensor[tensor_offset+x], packed_offset + x);
    }
  }
}

template <typename DataType, typename OpType>
__device__ static
typename std::enable_if<OpType::group == HaloTraversalOpGroup::BLOCK, void>::type
traverse_halo_opt_dim2_apply(DataType * __restrict__ tensor,
                             size_t tensor_offset_common,
                             size_t packed_offset_common,
                             int x_len, int y_len, OpType op) {
  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  for (int y = 0; y < y_len; y += blockDim.y) {
    for (int x = 0; x < x_len; x += blockDim.x) {
      // avoid reading beyond array boundary
      auto tensor_idx = min(x + threadIdx.x, x_len - 1)
          + min(y + threadIdx.y, y_len - 1) * x_len;
      op(tensor[tensor_offset_common+tensor_idx],
         packed_offset_common + x + y * x_len,
         tid, min(blockDim.x, x_len - x) * min(blockDim.y, y_len - y));
    }
  }
}


// ND: 5
// Traverse halo at dimension 1 with a known halo width
template <int ND, typename DataType, typename OpType,
          bool inner, Side side, int halo_width>
__device__ static void traverse_halo_opt_dim2(DataType * __restrict__ tensor,
                                              const Array<ND> &shape,
                                              OpType op) {
  constexpr int dim = 2;
  auto sample_idx = blockIdx.y;
  auto ch_offset = shape[0] * shape[1];
  auto sample_offset = ch_offset * shape[-2];
  int halo_idx = 0;
  if (side == Side::RHS) {
    if (inner) {
      halo_idx += (int)(shape[dim]) - halo_width * 2;
    } else {
      halo_idx += (int)(shape[dim]) - halo_width;
    }
  } else {
    if (inner) {
      halo_idx += halo_width;
    }
  }
  size_t offset_common = sample_idx * sample_offset + blockIdx.x * ch_offset;
  ch_offset *= gridDim.x;
  for (int ch_idx = blockIdx.x; ch_idx < shape[-2]; ch_idx += gridDim.x) {
    size_t packed_offset_common = offset_common * halo_width;
    size_t tensor_offset_common = offset_common * shape[2];
    tensor_offset_common += shape[0] * shape[1] * halo_idx;
#pragma unroll
    for (int hw = 0; hw < halo_width; ++hw) {
      traverse_halo_opt_dim2_apply(tensor, tensor_offset_common,
                                   packed_offset_common,
                                   shape[0], shape[1], op);
      packed_offset_common += shape[0] * shape[1];
      tensor_offset_common += shape[0] * shape[1];
    }
    offset_common += ch_offset;
  }
}

template <typename OpType>
__device__ static
typename std::enable_if<OpType::has_pre_grid, void>::type
traverse_halo_pre(OpType op) {
  if (gridDim.x * gridDim.y * gridDim.z == 1) {
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      op.pre();
    }
    __syncthreads();
  } else {
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 &&
        blockIdx.y == 0) {
      op.pre();
    }
    cooperative_groups::this_grid().sync();
  }
}

template <typename OpType>
__device__ static
typename std::enable_if<!OpType::has_pre_grid, void>::type
traverse_halo_pre(OpType op) {}

template <typename OpType>
__device__ static
typename std::enable_if<OpType::has_post_grid, void>::type
traverse_halo_post(OpType op) {
  if (gridDim.x * gridDim.y * gridDim.z == 1) {
    __syncthreads();
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      op.post();
    }
  } else {
    cooperative_groups::this_grid().sync();
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 &&
        blockIdx.y == 0) {
      op.post();
    }
  }
}

template <typename OpType>
__device__ static
typename std::enable_if<!OpType::has_post_grid, void>::type
traverse_halo_post(OpType op) {}

// Traversal with a known halo dimension
template <int ND, typename DataType, typename OpType, bool inner,
          Side side, int dim, int halo_width>
__global__ static void traverse_halo_opt(DataType * __restrict__ tensor, Array<ND> shape,
                                         OpType op) {
  traverse_halo_pre(op);
  if (dim == 0) {
    traverse_halo_opt_dim0<ND, DataType, OpType, inner, side>
        (tensor, shape, halo_width, op);
  } else if (dim == 1) {
    traverse_halo_opt_dim1<ND, DataType, OpType, inner, side, halo_width>(
        tensor, shape, op);
  } else if (dim == 2) {
    traverse_halo_opt_dim2<ND, DataType, OpType, inner, side, halo_width>(
        tensor, shape, op);
  }
  traverse_halo_post(op);
}

// Specialization with halo width
template <int ND, typename DataType, typename OpType,
          bool inner, Side side, int dim>
inline void traverse_halo_opt(DataType *tensor, Array<ND> shape,
                              int halo_width, OpType op,
                              h2::gpu::DeviceStream s, dim3 gsize, dim3 bsize) {

#define CALL(HW)                                                        \
  if ((OpType::has_post_grid || OpType::has_pre_grid) && num_blocks > 1) { \
    void *args[3] = {&tensor, &shape, &op};                             \
    DISTCONV_CHECK_GPU(GPU_LAUNCH_COOP_KERNEL(                          \
        (const void *)(traverse_halo_opt<ND, DataType, OpType, inner,   \
                       side, dim, HW>),                                 \
        gsize, bsize, (void **)args, 0, s));                            \
  } else {                                                              \
    traverse_halo_opt<ND, DataType, OpType, inner, side, dim, HW>       \
        <<<gsize, bsize, 0, s>>>(tensor, shape, op);                    \
    auto err = GPU_LAST_ERROR();                                        \
    if (err != GPU_SUCCESS) {                                           \
      util::MPIPrintStreamError()                                       \
          << "Lauch error: "                                            \
          << gsize.x << "x" << gsize.y << "x" << gsize.z                \
          << ", " << bsize.x << "x" << bsize.y << "x" << bsize.z;       \
      DISTCONV_CHECK_GPU(err);                                          \
    }                                                                   \
  }

  int num_blocks = gsize.x * gsize.y * gsize.z;

  switch (halo_width) {
    case 1:
      CALL(1);
      break;
    case 2:
      CALL(2);
      break;
    case 3:
      CALL(3);
      break;
    case 4:
      CALL(4);
      break;
      // Disable larger halo width to reduce compilation time
#if 0
    case 5:
      CALL(5);
      break;
    case 6:
      CALL(6);
      break;
    case 7:
      CALL(7);
      break;
#endif
    default:
      // In order to make the performance more consistent, make it
      // fail if the halo width is not included in the above cases.
      //traverse_halo_opt<ND, DataType, OpType, inner, side, dim>
      //<<<gsize, bsize, 0, s>>>(tensor, shape, halo_width, op);
      util::MPIRootPrintStreamError() << "Unsupported halo size: " << halo_width;
      throw std::exception();
  }
#undef CALL
}

inline int get_max_block_dimension() {
  int dim = 512; // default
  auto env = std::getenv("DISTCONV_HALO_TRAVERSAL_MAX_BLOCK_SIZE");
  if (env) {
    dim = std::stoi(std::string(env));
  }
  return dim;
}

// ND: 4, 5
// Use 2-way vector type if possible
template <int ND, typename DataType, typename OpType,
          bool inner, Side side, int dim>
inline void traverse_halo_opt(DataType *tensor, Array<ND> shape,
                              int halo_width, OpType op,
                              h2::gpu::DeviceStream s) {
  int vector_width = 1;
  if (dim >= 1) {
    if (shape[0] % 4 == 0) {
      // vector2 seems better than vector4.
      vector_width = 2;
    } else if (shape[0] % 2 == 0) {
      vector_width = 2;
    }
    shape[0] /= vector_width;
  } else if (dim == 0) {
    if (halo_width % 4 == 0 && shape[0] % 4 == 0) {
      // vector2 seems better than vector4.
      vector_width = 2;
    } else if (halo_width % 2 == 0 && shape[0] % 2 == 0) {
      vector_width = 2;
    }
    halo_width /= vector_width;
    shape[0] /= vector_width;
  } else {
    util::MPIRootPrintStreamError() << "Invalid dimension";
    throw std::exception();
  }

  dim3 bsize(1);
  size_t max_dim = get_max_block_dimension();
  if (dim == 0) {
    bsize.x = halo_width;
    bsize.y = std::min(shape[1], max_dim / bsize.x);
  } else if (dim == 1) {
    // The block-based traverse_halo_opt_dim1_5d_apply assumes one
    // thread block completely covers the x dimension.
    if (ND == 5) assert_always(shape[0] <= max_dim);
    bsize.x = std::min(shape[0], max_dim);
    if (ND == 5) bsize.y = std::min(shape[2], max_dim / bsize.x);
  } else if (dim == 2) {
    bsize.x = std::min(shape[0], max_dim);
    if (ND == 5) bsize.y = std::min(shape[1], max_dim / bsize.x);
  }
  constexpr int max_channel_per_block = 512;
  dim3 gsize(util::ceil(int(shape[-2]), max_channel_per_block), shape[-1]);
  // Compiling for additional vector width makes compilation time
  // further longer
  assert_always(vector_width == 2 || vector_width == 1);
  if (vector_width == 2) {
    using VecType = typename util::GetVectorType<DataType, 2>::type;
    traverse_halo_opt<ND, VecType, OpType, inner, side, dim>(
        (VecType*)tensor, shape, halo_width,
        op, s, gsize, bsize);
  } else {
    traverse_halo_opt<ND, DataType, OpType, inner, side, dim>(
        tensor, shape, halo_width, op, s, gsize, bsize);
  }
}

// ND: 4, 5
// Specialization with dimension
template <int ND, typename DataType, typename OpType,
          bool inner, Side side>
inline void traverse_halo_opt(DataType *tensor, Array<ND> shape,
                              int dim, int halo_width,
                              OpType op, h2::gpu::DeviceStream s) {
  if (dim == 0) {
    traverse_halo_opt<ND, DataType, OpType, inner, side, 0>(
        tensor, shape, halo_width, op, s);
  } else if (dim == 1) {
    traverse_halo_opt<ND, DataType, OpType, inner, side, 1>(
        tensor, shape, halo_width, op, s);
  } else {
    assert_eq(dim, 2);
    traverse_halo_opt<ND, DataType, OpType, inner, side, 2>(
        tensor, shape, halo_width, op, s);
  }
}

// ND: 4, 5
// Specialization with direction
template <int ND, typename DataType, typename OpType,
          bool inner>
inline void traverse_halo_opt(DataType *tensor, Array<ND> shape,
                              int dim, Side side,
                              int halo_width, OpType op,
                              h2::gpu::DeviceStream s) {
  if (side == Side::RHS) {
    traverse_halo_opt<ND, DataType, OpType, inner, Side::RHS>(
        tensor, shape, dim, halo_width, op, s);
  } else {
    traverse_halo_opt<ND, DataType, OpType, inner, Side::LHS>(
        tensor, shape, dim, halo_width, op, s);
  }
}

// ND: 4, 5
// Specialization with inner/outer
template <int ND, typename DataType, typename OpType>
inline void traverse_halo_opt(DataType *tensor, Array<ND> shape,
                              int dim, Side side,
                              int halo_width, bool inner,
                              OpType op, h2::gpu::DeviceStream s) {
  if (inner) {
    traverse_halo_opt<ND, DataType, OpType, true>(
        tensor, shape, dim, side, halo_width, op, s);
  } else {
    traverse_halo_opt<ND, DataType, OpType, false>(
        tensor, shape, dim, side, halo_width, op, s);
  }
}

} // namespace internal

template <typename Tensor, typename OpType>
void TraverseHalo(Tensor &tensor, int dim,
                  Side side, int halo_width,
                  bool inner, OpType op,
                  h2::gpu::DeviceStream s) {
  // ConstDataType is const DataType if the operation only reads the tensor.
  using ConstDataType = std::conditional_t<OpType::modifies_tensor,
                                           typename Tensor::data_type,
                                           typename Tensor::const_data_type>;
  const int num_dims = tensor.get_num_dims();
  int available_halo_width = tensor.get_halo_width(dim);
  assert_always(halo_width <= available_halo_width);
  if (inner) {
    assert_always(tensor.get_local_shape()[dim] >= (size_t)halo_width);
  }
  if (halo_width == 0) {
    // No halo region attached
    return;
  }
  if (num_dims == 4 && (dim == 0 || dim == 1)) {
    internal::traverse_halo_opt<4, ConstDataType, OpType>(
        static_cast<ConstDataType*>(tensor.get_buffer()),
        tensor.get_local_real_shape(),
        dim, side, halo_width,
        inner, op, s);
  } else if (num_dims == 5 && (dim == 0 || dim ==1 || dim == 2)) {
    internal::traverse_halo_opt<5, ConstDataType, OpType>(
        static_cast<ConstDataType*>(tensor.get_buffer()),
        tensor.get_local_real_shape(),
        dim, side, halo_width,
        inner, op, s);
  } else {
    auto local_real_shape = tensor.get_local_real_shape();
    local_real_shape[dim] = halo_width;
    auto num_halo_points = local_real_shape.get_size();
    int block_size = 256;
    int grid_size = (num_halo_points + block_size - 1) / block_size;
    // Block-based operation not supported
    assert_always(OpType::group == HaloTraversalOpGroup::THREAD);
    internal::traverse_halo_generic<ConstDataType, OpType>(
        static_cast<ConstDataType*>(tensor.get_buffer()),
        tensor.get_local_real_shape(),
        dim, side, inner, halo_width, num_halo_points, op,
        num_dims, grid_size, block_size, s);
  }
}

template <typename Tensor, typename OpType>
void TraverseHalo(Tensor &tensor, int dim,
                  Side side,
                  bool inner, OpType op,
                  h2::gpu::DeviceStream s) {
  TraverseHalo(tensor, dim, side,
               tensor.get_distribution().get_overlap()[dim],
               inner, op, s);
  return;
}

template <typename Tensor, typename OpType>
void TraverseHalo(Tensor &tensor, int dim,
                  int width_rhs, int width_lhs,
                  bool inner,
                  OpType op, h2::gpu::DeviceStream s) {
  TraverseHalo(tensor, dim, width_rhs, Side::RHS,
               inner, op, s);
  TraverseHalo(tensor, dim, width_lhs, Side::LHS,
               inner, op, s);
}

template <typename Tensor, typename OpType>
void TraverseHalo(Tensor &tensor, int dim,
                  bool inner,
                  OpType op, h2::gpu::DeviceStream s) {
  TraverseHalo(tensor, dim, Side::RHS, inner, op, s);
  TraverseHalo(tensor, dim, Side::LHS, inner, op, s);
}

} // namespace tensor
} // namespace distconv

#undef GPU_SUCCESS
#undef GPU_LAUNCH_COOP_KERNEL
#undef GPU_LAST_ERROR
