#pragma once

#include "distconv_config.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/runtime.hpp"
#include "distconv/runtime_cuda.hpp"

#include <cstdlib>
#include <cassert>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

#include "nvToolsExt.h"

#define DISTCONV_CHECK_CUDA(cuda_call)                                  \
  do {                                                                  \
    const cudaError_t cuda_status = cuda_call;                          \
    if (cuda_status != cudaSuccess) {                                   \
      std::cerr << "CUDA error: " << cudaGetErrorString(cuda_status) << "\n"; \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define DISTCONV_CHECK_CUBLAS(cublas_call)                              \
  do {                                                                  \
    const cublasStatus_t cublas_status = cublas_call;                   \
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {                       \
      std::cerr << "CUBLAS error";                                      \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << "\n";  \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

#define DISTCONV_CUDA_MALLOC(p, s)                              \
  distconv::util::cuda_malloc((void**)p, s, __FILE__, __LINE__)

namespace distconv {
namespace util {

int get_number_of_gpus();
int get_local_rank();
int get_local_size();

int choose_gpu();

std::ostream &operator<<(std::ostream &os, const cudaPitchedPtr &p);
std::ostream &operator<<(std::ostream &os, const cudaPos &p);
std::ostream &operator<<(std::ostream &os, const cudaMemcpy3DParms &p);

cudaError_t cuda_malloc(void **ptr, size_t size,
                        const char *file_name=nullptr, int linum=0);

void wait_stream(cudaStream_t master, cudaStream_t follower);
void wait_stream(cudaStream_t master, cudaStream_t *followers, int num_followers);
void sync_stream(cudaStream_t s1, cudaStream_t s2);

cudaStream_t create_priority_stream();

struct Clock {
  cudaStream_t m_s;
  cudaEvent_t m_ev1;
  cudaEvent_t m_ev2;
  Clock(cudaStream_t s): m_s(s) {
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_ev1));
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_ev2));
  }
  Clock(const Clock &c): Clock(c.m_s) {}
  Clock &operator=(const Clock &c) {
    m_s = c.m_s;
    return *this;
  }
  ~Clock() {
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_ev1));
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_ev2));
  }
  void start() {
    DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev1, m_s));
  }
  void stop() {
    DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev2, m_s));
  }
  float get_time() {
    DISTCONV_CHECK_CUDA(cudaEventSynchronize(m_ev2));
    float elapsed = 0;
    DISTCONV_CHECK_CUDA(cudaEventElapsedTime(&elapsed, m_ev1, m_ev2));
    return elapsed;
  }
};

inline void nvtx_push(const char *name) {
  if (get_config().m_nvtx) {
    nvtxRangePushA(name);
  }
}

inline void nvtx_pop() {
  if (get_config().m_nvtx) {
    nvtxRangePop();
  }
}

#define LIST_OF_VECTOR2_TYPES              \
  VECTOR_TYPE_OP(int, int2, 2)             \
  VECTOR_TYPE_OP(long, long2, 2)           \
  VECTOR_TYPE_OP(float, float2, 2)         \
  VECTOR_TYPE_OP(double, double2, 2)

#define LIST_OF_VECTOR4_TYPES              \
  VECTOR_TYPE_OP(int, int4, 4)             \
  VECTOR_TYPE_OP(long, long4, 4)           \
  VECTOR_TYPE_OP(float, float4, 4)         \
  VECTOR_TYPE_OP(double, double4, 4)

#define LIST_OF_VECTOR_TYPES                    \
  LIST_OF_VECTOR2_TYPES                         \
  LIST_OF_VECTOR4_TYPES

template <typename DataType, int x>
struct GetVectorType {
  using type = typename std::conditional<
    std::is_const<DataType>::value,
    typename std::add_const<typename GetVectorType<typename std::remove_const<DataType>::type, x>::type>::type,
    typename GetVectorType<typename std::remove_const<DataType>::type, x>::type>
      ::type;
};

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  struct GetVectorType<B, W> {                  \
    using type = V;                             \
  };

LIST_OF_VECTOR_TYPES

#undef VECTOR_TYPE_OP

template <typename T>
struct IsVectorType {
  using NonConstT = typename std::remove_const<T>::type;
  static constexpr bool value =
      std::is_same<NonConstT, int2>::value ||
      std::is_same<NonConstT, long2>::value ||
      std::is_same<NonConstT, float2>::value ||
      std::is_same<NonConstT, double2>::value ||
      std::is_same<NonConstT, int4>::value ||
      std::is_same<NonConstT, long4>::value ||
      std::is_same<NonConstT, float4>::value ||
      std::is_same<NonConstT, double4>::value;
};

template <typename T>
struct GetVectorWidth {
  static constexpr int width  = 1;
};

#define VECTOR_TYPE_OP(B, V, W)             \
  template <>                               \
  struct GetVectorWidth<V> {                \
    static constexpr int width  = W;        \
  };

LIST_OF_VECTOR_TYPES

#undef VECTOR_TYPE_OP

#ifdef __NVCC__

template <typename T>
inline __device__ T max(T x, T y) {
  return (x > y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  inline __device__ V max<V>(V x, V y) {        \
    V z;                                        \
    z.x = max(x.x, y.x);                        \
    z.y = max(x.y, y.y);                        \
    return z;                                   \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  inline __device__ V max<V>(V x, V y) {        \
    V z;                                        \
    z.x = max(x.x, y.x);                        \
    z.y = max(x.y, y.y);                        \
    z.z = max(x.z, y.z);                        \
    z.w = max(x.w, y.w);                        \
    return z;                                   \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
inline __device__ T min(T x, T y) {
  return (x < y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  inline __device__ V min<V>(V x, V y) {        \
    V z;                                        \
    z.x = min(x.x, y.x);                        \
    z.y = min(x.y, y.y);                        \
    return z;                                   \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  inline __device__ V min<V>(V x, V y) {        \
    V z;                                        \
    z.x = min(x.x, y.x);                        \
    z.y = min(x.y, y.y);                        \
    z.z = min(x.z, y.z);                        \
    z.w = min(x.w, y.w);                        \
    return z;                                   \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif // __NVCC__

} // namespace util

#ifdef __NVCC__
#define VECTOR_TYPE_OP(B, V, W)                         \
  inline __device__ void operator+=(V &x, V y) {        \
    x.x += y.x;                                         \
    x.y += y.y;                                         \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                         \
  inline __device__ void operator+=(V &x, V y) {        \
    x.x += y.x;                                         \
    x.y += y.y;                                         \
    x.z += y.z;                                         \
    x.w += y.w;                                         \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif // __NVCC__

} // namespace distconv

#undef LIST_OF_VECTOR_TYPES
