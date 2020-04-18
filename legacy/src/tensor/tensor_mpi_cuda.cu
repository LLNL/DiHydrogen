#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/tensor/halo_cuda.hpp"
#include "distconv/tensor/algorithms/transform_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_mpi.hpp"

#include <cuda_runtime.h>

namespace distconv {
namespace tensor {

namespace internal {

template <typename DataType>
struct ClearHaloFunctor {
  using Vec2 = typename util::GetVectorType<DataType, 2>::type;
  using Vec4 = typename util::GetVectorType<DataType, 4>::type;
  static constexpr HaloTraversalOpGroup group = HaloTraversalOpGroup::THREAD;
  static constexpr bool has_pre_grid = false;
  static constexpr bool has_post_grid = false;
  static constexpr bool modifies_tensor = true;
  ClearHaloFunctor() {}
  __device__ void operator()(DataType &x, size_t) {
    x = DataType(0);
  }
  __device__ void operator()(Vec2 &x, size_t) {
    x.x = DataType(0);
    x.y = DataType(0);
  }
  __device__ void operator()(Vec4 &x, size_t) {
    x.x = DataType(0);
    x.y = DataType(0);
    x.z = DataType(0);
    x.w = DataType(0);
  }
};

template <typename DataType>
struct ScaleFunctor {
  ScaleFunctor(DataType v): m_v(v) {}
  __device__ void operator()(DataType &x) {
    x *= m_v;
  }
  DataType m_v;
};

template <typename DataType1, typename DataType2>
struct CastFuctor {
  __device__ void operator()(DataType1 &x, const DataType2 &y) {
    x = static_cast<DataType1>(y);
  }
};

template <typename DataType1, typename DataType2>
struct CastScaleBiasFuctor {
  CastScaleBiasFuctor(const DataType1 alpha, const DataType1 beta)
      : m_alpba(alpha), m_beta(beta) {}
  __device__ void operator()(DataType1 &x, const DataType2 &y) {
    x = m_alpba*static_cast<DataType1>(y)+m_beta;
  }
  DataType1 m_alpba, m_beta;
};

} // namespace internal

#define DEFINE_CLEAR_HALO(TYPE)                                         \
  template <>                                                           \
  void TensorImplHelper<TYPE, CUDAAllocator>::clear_halo(int dim, cudaStream_t s) { \
    TraverseHalo<TensorImplType::TensorType, internal::ClearHaloFunctor<TYPE>>( \
        *(m_impl.get_tensor()), dim, false, internal::ClearHaloFunctor<TYPE>(), s); \
  }

DEFINE_CLEAR_HALO(float)
DEFINE_CLEAR_HALO(double)
DEFINE_CLEAR_HALO(int)

#undef DEFINE_CLEAR_HALO

#define DEFINE_SCALE(TYPE)                                              \
  template <>                                                           \
  void TensorImplHelper<TYPE, CUDAAllocator>::scale(TYPE v, cudaStream_t s) { \
    Transform(*(m_impl.get_tensor()), internal::ScaleFunctor<TYPE>(v), s); \
  }

DEFINE_SCALE(float)
DEFINE_SCALE(double)
DEFINE_SCALE(int)

#undef DEFINE_SCALE

#define DEFINE_CAST(T1, T2) \
  template <>                                                           \
  int Cast<T1, T2>(Tensor<T1, LocaleMPI, CUDAAllocator> &t_dest,        \
                   const Tensor<T2, LocaleMPI, CUDAAllocator> &t_src,   \
                   cudaStream_t stream) {                               \
    Transform(t_dest, t_src, internal::CastFuctor<T1, T2>(),            \
              stream);                                                  \
    return 0;                                                           \
  }
DEFINE_CAST(float, short)
DEFINE_CAST(float, unsigned short)
DEFINE_CAST(double, short)
DEFINE_CAST(double, unsigned short)
#undef DEFINE_CAST

#define DEFINE_CAST_SCALE_BIAS(T1, T2) \
  template <>                                                                    \
  int CastScaleBias<T1, T2>(Tensor<T1, LocaleMPI, CUDAAllocator> &t_dest,        \
                            const Tensor<T2, LocaleMPI, CUDAAllocator> &t_src,   \
                            const T1 alpha,                                      \
                            const T1 beta,                                       \
                            cudaStream_t stream) {                               \
    Transform(t_dest, t_src, internal::CastScaleBiasFuctor<T1, T2>(alpha, beta), \
              stream);                                                           \
    return 0;                                                                    \
  }
DEFINE_CAST_SCALE_BIAS(float, float)
DEFINE_CAST_SCALE_BIAS(float, short)
DEFINE_CAST_SCALE_BIAS(float, unsigned short)
DEFINE_CAST_SCALE_BIAS(double, double)
DEFINE_CAST_SCALE_BIAS(double, short)
DEFINE_CAST_SCALE_BIAS(double, unsigned short)
#undef DEFINE_CAST_SCALE_BIAS


namespace internal {

template <typename DataType1, typename DataType2>
__device__ __forceinline__ void assign(DataType1 &t1, DataType2 &t2) {
}

template <typename DataType1, typename DataType2>
__device__ __forceinline__ void assign(const DataType1 &t1, DataType2 &t2) {
  t2 = t1;
}

template <typename DataType1, typename DataType2>
__device__ __forceinline__ void assign(DataType1 &t1, const DataType2 &t2) {
  t1 = t2;
}

template <int ND, int INNER_DIM, typename DataType1, typename DataType2,
          bool is_concat>
__global__ void concat_or_slice_kernel(
    DataType1 *dst, Array<ND> dst_shape, Array<ND> dst_strides,
    DataType2 *src1, Array<ND> src1_shape, Array<ND> src1_strides,
    DataType2 *src2, Array<ND> src2_shape, Array<ND> src2_strides,
    int concat_dim) {
  // NOTE: For simplicity, dimension of concat_dim is assumed to be traversed by
  // different thread blocks.
  const int tid = threadIdx.x;
  int bid = blockIdx.x;
  const int block_size = blockDim.x;
  DataType2 *src = nullptr;
  Array<ND> src_strides;
  Array<ND> src_block_idx;
  Array<ND> dst_block_idx;
#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    auto idx = bid % dst_shape[i];
    dst_block_idx[i] = idx;
    bid /= dst_shape[i];
    src_block_idx[i] = idx;
    if (i == concat_dim) {
      if (idx < src1_shape[i]) {
        src = src1;
        src_strides = src1_strides;
      } else {
        src = src2;
        src_strides = src2_strides;
        src_block_idx[i] = idx - src1_shape[i];
      }
    }
  }

#pragma unroll
  for (int i = INNER_DIM + 1; i < ND; ++i) {
    dst += dst_block_idx[i] * dst_strides[i];
    src += src_block_idx[i] * src_strides[i];
  }

  // Assume the region a thread block traverses has the same shape
  // between dst and src
  int inner_size = 1;
#pragma unroll
  for (int i = 0; i <= INNER_DIM; ++i) {
    inner_size *= dst_shape[i];
  }
  for (int inner_idx = tid; inner_idx < inner_size; inner_idx += block_size) {
    int dst_offset = 0;
    int src_offset = 0;
    int inner_idx_i = inner_idx;
#pragma unroll
    for (int j = 0; j <= INNER_DIM; ++j) {
      int idx_j = inner_idx_i % dst_shape[j];
      dst_offset += dst_strides[j] * idx_j;
      src_offset += src_strides[j] * idx_j;
      inner_idx_i /= dst_shape[j];
    }
    assign(dst[dst_offset], src[src_offset]);
  }
}

template <bool B, typename T>
struct AddConstIf;

template <typename T>
struct AddConstIf<true, T> {
  using type = typename std::add_const<T>::type;
};

template <typename T>
struct AddConstIf<false, T> {
  using type = T;
};


template <typename DataType, bool IS_CONCAT>
int ConcatenateOrSlice(
    typename AddConstIf<!IS_CONCAT, Tensor<DataType, LocaleMPI, CUDAAllocator>>::type &t_dest,
    typename AddConstIf<IS_CONCAT, Tensor<DataType, LocaleMPI, CUDAAllocator>>::type &t_src1,
    typename AddConstIf<IS_CONCAT, Tensor<DataType, LocaleMPI, CUDAAllocator>>::type &t_src2,
    cudaStream_t s) {
  const int nd = t_dest.get_num_dims();
  int block_dim = 256; // tunable

  int concat_dim = -1;
  for (int i = 0; i < nd; ++i) {
    auto dest_dim = t_dest.get_shape()[i];
    auto src1_dim = t_src1.get_shape()[i];
    auto src2_dim = t_src2.get_shape()[i];
    if (dest_dim == src1_dim && dest_dim == src2_dim) {
      // this is not concat dim
      continue;
    }
    assert_always(dest_dim == src1_dim + src2_dim);
    concat_dim = i;
    break;
  }

  // TODO: only works for U-Net. Concat on channel dim
  assert_always(concat_dim == nd - 2);

  using DataType1 = typename AddConstIf<!IS_CONCAT, DataType>::type;
  using DataType2 = typename AddConstIf<IS_CONCAT, DataType>::type;

#define CALL_KERNEL(ND, INNER_DIM)  do {                                \
    assert_always(concat_dim > INNER_DIM);                              \
    int grid_dim = 1;                                                   \
    for (int i = INNER_DIM + 1; i < ND; ++i) {                          \
      grid_dim *= t_dest.get_local_shape()[i];                          \
    }                                                                   \
    concat_or_slice_kernel<ND, INNER_DIM, DataType1, DataType2, IS_CONCAT> \
          <<<grid_dim, block_dim, 0, s>>>(                              \
              t_dest.get_base_ptr(), Array<ND>(t_dest.get_local_shape()), \
              Array<ND>(t_dest.get_strides()),                          \
              t_src1.get_base_ptr(), Array<ND>(t_src1.get_local_shape()), \
              Array<ND>(t_src1.get_strides()),                          \
              t_src2.get_base_ptr(), Array<ND>(t_src2.get_local_shape()), \
              Array<ND>(t_src2.get_strides()),                          \
              concat_dim);                                              \
  } while (0)

  switch (nd) {
    case 3:
      CALL_KERNEL(3, 1);
      break;
    case 4:
      CALL_KERNEL(4, 1);
      break;
    case 5:
      // Needs more robust tuning
      CALL_KERNEL(5, 1);
      break;
    default:
      throw std::exception();
  }
#undef CALL_KERNEL
  return 0;
}
} // namespace internal

template <typename DataType>
int Concatenate(Tensor<DataType, LocaleMPI, CUDAAllocator> &t_dest,
                const Tensor<DataType, LocaleMPI, CUDAAllocator> &t_src1,
                const Tensor<DataType, LocaleMPI, CUDAAllocator> &t_src2,
                cudaStream_t s) {
  return internal::ConcatenateOrSlice<DataType, true>(
      t_dest, t_src1, t_src2, s);
}

template <typename DataType>
int Slice(Tensor<DataType, LocaleMPI, CUDAAllocator> &t_dest1,
          Tensor<DataType, LocaleMPI, CUDAAllocator> &t_dest2,
          const Tensor<DataType, LocaleMPI, CUDAAllocator> &t_src,
          cudaStream_t s) {
  return internal::ConcatenateOrSlice<DataType, false>(
      t_src, t_dest1, t_dest2, s);
}

#define DEFINE_CONCATENATE(TYPE)                                        \
  template                                                              \
  int Concatenate<TYPE>(Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_dest, \
      const Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_src1,             \
                  const Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_src2, \
                        cudaStream_t s);                                \
  template                                                              \
  int Slice<TYPE>(Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_dest1,      \
                  Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_dest2,      \
                  const Tensor<TYPE, LocaleMPI, CUDAAllocator> &t_src,  \
                  cudaStream_t s);
DEFINE_CONCATENATE(float)
DEFINE_CONCATENATE(double)
DEFINE_CONCATENATE(int)
DEFINE_CONCATENATE(long)
#undef DEFINE_CONCATENATE

} // namespace tensor
} // namespace distconv
