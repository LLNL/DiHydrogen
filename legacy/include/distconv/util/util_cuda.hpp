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
#include <cfloat>

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

#ifdef __CUDACC__
template <typename T>
__device__ __forceinline__ T atomic_add(T* address, T value)
{
  return atomicAdd(address, value);
}

// Handle fp16
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700 && __CUDA_ARCH__ >= 530
#include <cuda_fp16.h>
__device__ __forceinline__ __half atomic_add(__half* address, __half val)
{
  unsigned int* address_as_uint = (unsigned int*) address;
  unsigned int old = *address_as_uint;
  __half* old_as_half = (__half*) &old;
  unsigned int assumed;
  unsigned int updated;
  __half* updated_as_half = (__half*) &updated;
  do {
    assumed = old;
    updated = old;
    *updated_as_half += val;
    old = atomicCAS(address_as_uint, assumed, updated);
  } while (assumed != old);
  return *old_as_half;
}
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700 && __CUDA_ARCH__ >= 530

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ __forceinline__ double atomic_add(double* address, double val)
{
  unsigned long long int* address_as_ull =
    (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                                         __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif // defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
#endif // __CUDACC__

namespace distconv {
namespace util {

void check_for_device_runtime_error();

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
  if (get_config().profiling) {
    nvtxRangePushA(name);
  }
}

inline void nvtx_pop() {
  if (get_config().profiling) {
    nvtxRangePop();
  }
}

#define LIST_OF_ELEMENT_TYPES                   \
  ELEMENT_TYPE_OP(int)                          \
  ELEMENT_TYPE_OP(long)                         \
  ELEMENT_TYPE_OP(float)                        \
  ELEMENT_TYPE_OP(double)

#define LIST_OF_VECTOR2_TYPES                   \
  VECTOR_TYPE_OP(int, int2, 2)                  \
  VECTOR_TYPE_OP(long, long2, 2)                \
  VECTOR_TYPE_OP(float, float2, 2)              \
  VECTOR_TYPE_OP(double, double2, 2)

#define LIST_OF_VECTOR4_TYPES                   \
  VECTOR_TYPE_OP(int, int4, 4)                  \
  VECTOR_TYPE_OP(long, long4, 4)                \
  VECTOR_TYPE_OP(float, float4, 4)              \
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

template <typename VectorType>
struct GetElementType {
  using type = typename std::conditional<
    std::is_const<VectorType>::value,
    typename std::add_const<typename GetElementType<typename std::remove_const<VectorType>::type>::type>::type,
    typename GetElementType<typename std::remove_const<VectorType>::type>::type>
      ::type;
};

#define VECTOR_TYPE_OP(B, V, W)                 \
  template <>                                   \
  struct GetVectorType<B, W> {                  \
    using type = V;                             \
  };                                            \
  template <>                                   \
  struct GetElementType<V> {                    \
    using type = B;                             \
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

template<typename B, typename V>
__device__ __forceinline__ V make_vector(B x);
template<typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y);
template<typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y, B z);
template<typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y, B z, B w);


#define VECTOR_TYPE_OP(B, V, W)                                         \
  template<>                                                            \
  __device__ __forceinline__ V make_vector<B, V>(B x) {                 \
    return make_##V(x, x);                                              \
  }                                                                     \
  template<>                                                            \
  __device__ __forceinline__ V make_vector<B, V>(B x, B y) {            \
    return make_##V(x, y);                                              \
  }
LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                         \
  template<>                                                            \
  __device__ __forceinline__ V make_vector<B, V>(B x) {                 \
    return make_##V(x, x, x, x);                                        \
  }                                                                     \
  template<>                                                            \
  __device__ __forceinline__ V make_vector<B, V>(B x, B y, B z, B w) {  \
    return make_##V(x, y, z, w);                                        \
  }
LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
__device__ __forceinline__ T max(T x, T y) {
  return (x > y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                                 \
  template <>                                                   \
  __device__ __forceinline__ V max<V>(V x, V y) {               \
    return make_vector<B, V>(max(x.x, y.x), max(x.y, y.y));     \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                 \
  template <>                                                   \
  __device__ __forceinline__ V max<V>(V x, V y) {               \
  return make_vector<B, V>(max(x.x, y.x), max(x.y, y.y),        \
                           max(x.z, y.z), max(x.w, y.w));       \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
__device__ __forceinline__ T min(T x, T y) {
  return (x < y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                                 \
  template <>                                                   \
  __device__ __forceinline__ V min<V>(V x, V y) {               \
    return make_vector<B, V>(min(x.x, y.x), min(x.y, y.y));     \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                 \
  template <>                                                   \
  __device__ __forceinline__ V min<V>(V x, V y) {               \
    return make_vector<B, V>(min(x.x, y.x), min(x.y, y.y),      \
                             min(x.z, y.z), min(x.w, y.w));     \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T> constexpr __device__ __forceinline__ T min();
template <> constexpr __device__ __forceinline__ float min<float>() {
  return FLT_MIN;
}
template <> constexpr __device__ __forceinline__ double min<double>() {
  return DBL_MIN;
}
template <typename T> constexpr __device__ __forceinline__ T max();
template <> constexpr __device__ __forceinline__ float max<float>() {
  return FLT_MAX;
}
template <> constexpr __device__ __forceinline__ double max<double>() {
  return DBL_MAX;
}

#define ELEMENT_TYPE_OP(B)                      \
  __device__ __forceinline__ B sum(B x) {       \
    return x;                                   \
  }
LIST_OF_ELEMENT_TYPES
#undef ELEMENT_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                 \
  __device__ __forceinline__ B sum(V x) {       \
    return x.x + x.y;                           \
  }
LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                 \
  __device__ __forceinline__ B sum(V x) {       \
    return x.x + x.y + x.z + x.w;               \
  }
LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif // __NVCC__

} // namespace util

#ifdef __NVCC__
#define VECTOR_TYPE_OP(B, V, W)                                 \
  __device__ __forceinline__ V operator+(V x, V y) {            \
    return util::make_vector<B, V>(x.x + y.x, x.y + y.y);       \
  }                                                             \
  __device__ __forceinline__ V operator-(V x, V y) {            \
    return util::make_vector<B, V>(x.x - y.x, x.y - y.y);       \
  }                                                             \
  __device__ __forceinline__ V operator*(V x, V y) {            \
    return util::make_vector<B, V>(x.x * y.x, x.y * y.y);       \
  }                                                             \
  __device__ __forceinline__ V operator/(V x, V y) {            \
    return util::make_vector<B, V>(x.x / y.x, x.y / y.y);       \
  }                                                             \
  __device__ __forceinline__ void operator+=(V &x, V y) {       \
    x.x += y.x;                                                 \
    x.y += y.y;                                                 \
  }                                                             \
  __device__ __forceinline__ void operator-=(V &x, V y) {       \
    x.x -= y.x;                                                 \
    x.y -= y.y;                                                 \
  }                                                             \
  __device__ __forceinline__ void operator*=(V &x, V y) {       \
    x.x *= y.x;                                                 \
    x.y *= y.y;                                                 \
  }                                                             \
  __device__ __forceinline__ void operator/=(V &x, V y) {       \
    x.x /= y.x;                                                 \
    x.y /= y.y;                                                 \
  }                                                             \
  __device__ __forceinline__ V operator+(V x, B y) {            \
    return util::make_vector<B, V>(x.x + y, x.y + y);           \
  }                                                             \
  __device__ __forceinline__ V operator-(V x, B y) {            \
    return util::make_vector<B, V>(x.x - y, x.y - y);           \
  }                                                             \
  __device__ __forceinline__ V operator*(V x, B y) {            \
    return util::make_vector<B, V>(x.x * y, x.y * y);           \
  }                                                             \
  __device__ __forceinline__ V operator/(V x, B y) {            \
    return util::make_vector<B, V>(x.x / y, x.y / y);           \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                         \
  __device__ __forceinline__ V operator+(V x, V y) {                    \
    return util::make_vector<B, V>(x.x + y.x, x.y + y.y,                \
                                   x.z + y.z, x.w + y.w);               \
  }                                                                     \
  __device__ __forceinline__ V operator-(V x, V y) {                    \
    return util::make_vector<B, V>(x.x - y.x, x.y - y.y,                \
                                   x.z - y.z, x.w - y.w);               \
  }                                                                     \
  __device__ __forceinline__ V operator*(V x, V y) {                    \
  return util::make_vector<B, V>(x.x * y.x, x.y * y.y,                  \
                                 x.z * y.z, x.w * y.w);                 \
  }                                                                     \
  __device__ __forceinline__ V operator/(V x, V y) {                    \
  return util::make_vector<B, V>(x.x / y.x, x.y / y.y,                  \
                                 x.z / y.z, x.w / y.w);                 \
  }                                                                     \
  __device__ __forceinline__ void operator+=(V &x, V y) {               \
    x.x += y.x;                                                         \
    x.y += y.y;                                                         \
    x.z += y.z;                                                         \
    x.w += y.w;                                                         \
  }                                                                     \
  __device__ __forceinline__ void operator-=(V &x, V y) {               \
    x.x -= y.x;                                                         \
    x.y -= y.y;                                                         \
    x.z -= y.z;                                                         \
    x.w -= y.w;                                                         \
  }                                                                     \
  __device__ __forceinline__ void operator*=(V &x, V y) {               \
    x.x *= y.x;                                                         \
    x.y *= y.y;                                                         \
    x.z *= y.z;                                                         \
    x.w *= y.w;                                                         \
  }                                                                     \
  __device__ __forceinline__ void operator/=(V &x, V y) {               \
    x.x /= y.x;                                                         \
    x.y /= y.y;                                                         \
    x.z /= y.z;                                                         \
    x.w /= y.w;                                                         \
  }                                                                     \
  __device__ __forceinline__ V operator+(V x, B y) {                    \
    return util::make_vector<B, V>(x.x + y, x.y + y, x.z + y, x.w + y); \
  }                                                                     \
  __device__ __forceinline__ V operator-(V x, B y) {                    \
    return util::make_vector<B, V>(x.x - y, x.y - y, x.z - y, x.w - y); \
  }                                                                     \
  __device__ __forceinline__ V operator*(V x, B y) {                    \
    return util::make_vector<B, V>(x.x * y, x.y * y, x.z * y, x.w * y); \
  }                                                                     \
  __device__ __forceinline__ V operator/(V x, B y) {                    \
    return util::make_vector<B, V>(x.x / y, x.y / y, x.z / y, x.w / y); \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif // __NVCC__

} // namespace distconv

#undef LIST_OF_VECTOR_TYPES
