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
#undef DEFINE_CAST_SCALE_BIAS

} // namespace tensor
} // namespace distconv
