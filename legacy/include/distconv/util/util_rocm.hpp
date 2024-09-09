////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "distconv_config.hpp"

#include "distconv/runtime.hpp"
#include "distconv/runtime_rocm.hpp"
#include "distconv/util/util_mpi.hpp"
#include "h2/gpu/runtime.hpp"

#include <cfloat>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <roctracer/roctx.h>

#define DISTCONV_CHECK_HIP(hip_call)                                           \
  do                                                                           \
  {                                                                            \
    const hipError_t status_distconv_check_hip = (hip_call);                   \
    if (status_distconv_check_hip != hipSuccess)                               \
    {                                                                          \
      std::cerr << "HIP error (" << hipGetErrorName(status_distconv_check_hip) \
                << ") at " << __FILE__ << ":" << __LINE__ << ": "              \
                << hipGetErrorString(status_distconv_check_hip) << std::endl;  \
      static_cast<void>(hipDeviceReset());                                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define DISTCONV_CHECK_ROCBLAS(rocblas_call)                                   \
  do                                                                           \
  {                                                                            \
    const rocblasStatus_t status_distconv_check_rocblas = (rocblas_call);      \
    if (status_distconv_check_rocblas != rocblas_status_success)               \
    {                                                                          \
      std::cerr << "ROCBLAS error at " << __FILE__ << ":" << __LINE__ << ": "  \
                << rocblas_status_to_string(status_distconv_check_rocblas)     \
                << std::endl;                                                  \
      static_cast<void>(hipDeviceReset());                                     \
      abort();                                                                 \
    }                                                                          \
  } while (0)

#define DISTCONV_HIP_MALLOC(p, s)                                              \
  DISTCONV_CHECK_HIP(                                                          \
    distconv::util::hip_malloc((void**) p, s, __FILE__, __LINE__))

#ifdef __HIPCC__
template <typename T>
__device__ __forceinline__ T atomic_add(T* address, T value)
{
  return atomicAdd(address, value);
}
#endif  // __HIPCC__

namespace distconv
{
namespace util
{

/** @brief Check if an error has been detected by the HIP runtime. */
void check_for_device_runtime_error();

/** @brief Get the node rank of this process.
 *
 *  This is technically launcher/MPI-dependent, but all cases I'm aware
 *  of map these consistently with global rank (i.e., in
 *  COMM_WORLD). That is, the process with lowest global rank on the
 *  node will have local rank 0.
 */
int get_local_rank();
/** @brief Get the number of processes on this node. */
int get_local_size();
/** @brief Suggest a device ID that should be chosen by this process.
 *
 *  This will generally be `get_local_rank % num_devices`, though this
 *  is subject to launcher effects, including plugins like mpibind.
 */
int choose_gpu();
/** @brief Get the number of GPUs available to this process. */
int get_num_gpus();

std::ostream& operator<<(std::ostream& os, const hipPitchedPtr& p);
std::ostream& operator<<(std::ostream& os, const hipPos& p);
std::ostream& operator<<(std::ostream& os, const hipMemcpy3DParms& p);

hipError_t hip_malloc(void** ptr,
                      size_t size,
                      const char* file_name = nullptr,
                      int linum = 0);

void wait_stream(hipStream_t master, hipStream_t follower);
void wait_stream(hipStream_t master, hipStream_t* followers, int num_followers);
void sync_stream(hipStream_t s1, hipStream_t s2);

hipStream_t create_priority_stream();

struct Clock
{
  hipStream_t m_s;
  hipEvent_t m_ev1;
  hipEvent_t m_ev2;
  Clock(hipStream_t s)
    : m_s{s}, m_ev1{h2::gpu::make_event()}, m_ev2{h2::gpu::make_event()}
  {}
  Clock(const Clock& c) : Clock(c.m_s) {}
  Clock& operator=(const Clock& c)
  {
    m_s = c.m_s;
    return *this;
  }
  ~Clock()
  {
    h2::gpu::destroy(m_ev2);
    h2::gpu::destroy(m_ev1);
  }
  void start() { DISTCONV_CHECK_HIP(hipEventRecord(m_ev1, m_s)); }
  void stop() { DISTCONV_CHECK_HIP(hipEventRecord(m_ev2, m_s)); }
  float get_time()
  {
    h2::gpu::sync(m_ev2);
    float elapsed = 0;
    DISTCONV_CHECK_HIP(hipEventElapsedTime(&elapsed, m_ev1, m_ev2));
    return elapsed;
  }
};

inline void profile_push(const char* name)
{
  if (get_config().profiling)
  {
    roctxRangePushA(name);
  }
}

inline void profile_pop()
{
  if (get_config().profiling)
  {
    roctxRangePop();
  }
}

#define LIST_OF_ELEMENT_TYPES                                                  \
  ELEMENT_TYPE_OP(int)                                                         \
  ELEMENT_TYPE_OP(long)                                                        \
  ELEMENT_TYPE_OP(float)                                                       \
  ELEMENT_TYPE_OP(double)

#define LIST_OF_VECTOR2_TYPES                                                  \
  VECTOR_TYPE_OP(int, int2, 2)                                                 \
  VECTOR_TYPE_OP(long, long2, 2)                                               \
  VECTOR_TYPE_OP(float, float2, 2)                                             \
  VECTOR_TYPE_OP(double, double2, 2)

#define LIST_OF_VECTOR4_TYPES                                                  \
  VECTOR_TYPE_OP(int, int4, 4)                                                 \
  VECTOR_TYPE_OP(long, long4, 4)                                               \
  VECTOR_TYPE_OP(float, float4, 4)                                             \
  VECTOR_TYPE_OP(double, double4, 4)

#define LIST_OF_VECTOR_TYPES                                                   \
  LIST_OF_VECTOR2_TYPES                                                        \
  LIST_OF_VECTOR4_TYPES

template <typename DataType, int x>
struct GetVectorType
{
  using type = typename std::conditional<
    std::is_const<DataType>::value,
    typename std::add_const<
      typename GetVectorType<typename std::remove_const<DataType>::type,
                             x>::type>::type,
    typename GetVectorType<typename std::remove_const<DataType>::type,
                           x>::type>::type;
};

template <typename VectorType>
struct GetElementType
{
  using type = typename std::conditional<
    std::is_const<VectorType>::value,
    typename std::add_const<typename GetElementType<
      typename std::remove_const<VectorType>::type>::type>::type,
    typename GetElementType<
      typename std::remove_const<VectorType>::type>::type>::type;
};

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  struct GetVectorType<B, W>                                                   \
  {                                                                            \
    using type = V;                                                            \
  };                                                                           \
  template <>                                                                  \
  struct GetElementType<V>                                                     \
  {                                                                            \
    using type = B;                                                            \
  };

LIST_OF_VECTOR_TYPES

#undef VECTOR_TYPE_OP

template <typename T>
struct IsVectorType
{
  using NonConstT = typename std::remove_const<T>::type;
  static constexpr bool value = std::is_same<NonConstT, int2>::value
                                || std::is_same<NonConstT, long2>::value
                                || std::is_same<NonConstT, float2>::value
                                || std::is_same<NonConstT, double2>::value
                                || std::is_same<NonConstT, int4>::value
                                || std::is_same<NonConstT, long4>::value
                                || std::is_same<NonConstT, float4>::value
                                || std::is_same<NonConstT, double4>::value;
};

template <typename T>
struct GetVectorWidth
{
  static constexpr int width = 1;
};

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  struct GetVectorWidth<V>                                                     \
  {                                                                            \
    static constexpr int width = W;                                            \
  };

LIST_OF_VECTOR_TYPES

#undef VECTOR_TYPE_OP

#ifdef __HIPCC__

template <typename B, typename V>
__device__ __forceinline__ V make_vector(B x);
template <typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y);
template <typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y, B z);
template <typename B, typename V>
__device__ __forceinline__ V make_vector(B x, B y, B z, B w);

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V make_vector<B, V>(B x)                          \
  {                                                                            \
    return make_##V(x, x);                                                     \
  }                                                                            \
  template <>                                                                  \
  __device__ __forceinline__ V make_vector<B, V>(B x, B y)                     \
  {                                                                            \
    return make_##V(x, y);                                                     \
  }
LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V make_vector<B, V>(B x)                          \
  {                                                                            \
    return make_##V(x, x, x, x);                                               \
  }                                                                            \
  template <>                                                                  \
  __device__ __forceinline__ V make_vector<B, V>(B x, B y, B z, B w)           \
  {                                                                            \
    return make_##V(x, y, z, w);                                               \
  }
LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
__device__ __forceinline__ T max(T x, T y)
{
  return (x > y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V max<V>(V x, V y)                                \
  {                                                                            \
    return make_vector<B, V>(max(x.x, y.x), max(x.y, y.y));                    \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V max<V>(V x, V y)                                \
  {                                                                            \
    return make_vector<B, V>(                                                  \
      max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w));             \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
__device__ __forceinline__ T min(T x, T y)
{
  return (x < y) ? x : y;
}

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V min<V>(V x, V y)                                \
  {                                                                            \
    return make_vector<B, V>(min(x.x, y.x), min(x.y, y.y));                    \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  template <>                                                                  \
  __device__ __forceinline__ V min<V>(V x, V y)                                \
  {                                                                            \
    return make_vector<B, V>(                                                  \
      min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), min(x.w, y.w));             \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

template <typename T>
constexpr __device__ __forceinline__ T min();
template <>
constexpr __device__ __forceinline__ float min<float>()
{
  return FLT_MIN;
}
template <>
constexpr __device__ __forceinline__ double min<double>()
{
  return DBL_MIN;
}
template <typename T>
constexpr __device__ __forceinline__ T max();
template <>
constexpr __device__ __forceinline__ float max<float>()
{
  return FLT_MAX;
}
template <>
constexpr __device__ __forceinline__ double max<double>()
{
  return DBL_MAX;
}

#define ELEMENT_TYPE_OP(B)                                                     \
  __device__ __forceinline__ B sum(B x)                                        \
  {                                                                            \
    return x;                                                                  \
  }
LIST_OF_ELEMENT_TYPES
#undef ELEMENT_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  __device__ __forceinline__ B sum(V x)                                        \
  {                                                                            \
    return x.x + x.y;                                                          \
  }
LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  __device__ __forceinline__ B sum(V x)                                        \
  {                                                                            \
    return x.x + x.y + x.z + x.w;                                              \
  }
LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif  // __HIPCC__

}  // namespace util

#ifdef __HIPCC__
#define VECTOR_TYPE_OP(B, V, W)                                                \
  __device__ __forceinline__ V operator+(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x + y.x, x.y + y.y);                      \
  }                                                                            \
  __device__ __forceinline__ V operator-(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x - y.x, x.y - y.y);                      \
  }                                                                            \
  __device__ __forceinline__ V operator*(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x * y.x, x.y * y.y);                      \
  }                                                                            \
  __device__ __forceinline__ V operator/(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x / y.x, x.y / y.y);                      \
  }                                                                            \
  __device__ __forceinline__ void operator+=(V& x, V y)                        \
  {                                                                            \
    x.x += y.x;                                                                \
    x.y += y.y;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator-=(V& x, V y)                        \
  {                                                                            \
    x.x -= y.x;                                                                \
    x.y -= y.y;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator*=(V& x, V y)                        \
  {                                                                            \
    x.x *= y.x;                                                                \
    x.y *= y.y;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator/=(V& x, V y)                        \
  {                                                                            \
    x.x /= y.x;                                                                \
    x.y /= y.y;                                                                \
  }                                                                            \
  __device__ __forceinline__ V operator+(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x + y, x.y + y);                          \
  }                                                                            \
  __device__ __forceinline__ V operator-(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x - y, x.y - y);                          \
  }                                                                            \
  __device__ __forceinline__ V operator*(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x * y, x.y * y);                          \
  }                                                                            \
  __device__ __forceinline__ V operator/(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x / y, x.y / y);                          \
  }

LIST_OF_VECTOR2_TYPES
#undef VECTOR_TYPE_OP

#define VECTOR_TYPE_OP(B, V, W)                                                \
  __device__ __forceinline__ V operator+(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(                                            \
      x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);                             \
  }                                                                            \
  __device__ __forceinline__ V operator-(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(                                            \
      x.x - y.x, x.y - y.y, x.z - y.z, x.w - y.w);                             \
  }                                                                            \
  __device__ __forceinline__ V operator*(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(                                            \
      x.x * y.x, x.y * y.y, x.z * y.z, x.w * y.w);                             \
  }                                                                            \
  __device__ __forceinline__ V operator/(V x, V y)                             \
  {                                                                            \
    return util::make_vector<B, V>(                                            \
      x.x / y.x, x.y / y.y, x.z / y.z, x.w / y.w);                             \
  }                                                                            \
  __device__ __forceinline__ void operator+=(V& x, V y)                        \
  {                                                                            \
    x.x += y.x;                                                                \
    x.y += y.y;                                                                \
    x.z += y.z;                                                                \
    x.w += y.w;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator-=(V& x, V y)                        \
  {                                                                            \
    x.x -= y.x;                                                                \
    x.y -= y.y;                                                                \
    x.z -= y.z;                                                                \
    x.w -= y.w;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator*=(V& x, V y)                        \
  {                                                                            \
    x.x *= y.x;                                                                \
    x.y *= y.y;                                                                \
    x.z *= y.z;                                                                \
    x.w *= y.w;                                                                \
  }                                                                            \
  __device__ __forceinline__ void operator/=(V& x, V y)                        \
  {                                                                            \
    x.x /= y.x;                                                                \
    x.y /= y.y;                                                                \
    x.z /= y.z;                                                                \
    x.w /= y.w;                                                                \
  }                                                                            \
  __device__ __forceinline__ V operator+(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x + y, x.y + y, x.z + y, x.w + y);        \
  }                                                                            \
  __device__ __forceinline__ V operator-(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x - y, x.y - y, x.z - y, x.w - y);        \
  }                                                                            \
  __device__ __forceinline__ V operator*(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x * y, x.y * y, x.z * y, x.w * y);        \
  }                                                                            \
  __device__ __forceinline__ V operator/(V x, B y)                             \
  {                                                                            \
    return util::make_vector<B, V>(x.x / y, x.y / y, x.z / y, x.w / y);        \
  }

LIST_OF_VECTOR4_TYPES
#undef VECTOR_TYPE_OP

#endif  // __HIPCC__

}  // namespace distconv

#undef LIST_OF_VECTOR_TYPES
