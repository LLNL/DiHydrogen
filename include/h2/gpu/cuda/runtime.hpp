////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cuda_runtime.h>

#include <tuple>

#include "h2/utils/Error.hpp"
#include "h2/meta/TypeList.hpp"


#define H2_CHECK_CUDA(CMD)                                              \
  do                                                                    \
  {                                                                     \
    const auto status_H2_CHECK_CUDA = (CMD);                            \
    if (status_H2_CHECK_CUDA != cudaSuccess)                            \
    {                                                                   \
      throw H2FatalException("CUDA error ",                             \
                             ::h2::gpu::error_name(status_H2_CHECK_CUDA), \
                             " (",                                      \
                             status_H2_CHECK_CUDA,                      \
                             "): ",                                     \
                             ::h2::gpu::error_string(status_H2_CHECK_CUDA)); \
    }                                                                   \
  } while (0)

namespace h2
{
namespace gpu
{

typedef cudaStream_t DeviceStream;
typedef cudaEvent_t DeviceEvent;
typedef cudaError_t DeviceError;

constexpr unsigned int max_grid_x = 2147483647;
constexpr unsigned int max_grid_y = 65535;
constexpr unsigned int max_grid_z = 65535;
constexpr unsigned int max_block_x = 1024;
constexpr unsigned int max_block_y = 1024;
constexpr unsigned int max_block_z = 64;
constexpr unsigned int max_threads_per_block = 1024;
constexpr unsigned int warp_size = 32;

inline bool ok(DeviceError status) noexcept
{
    return (status == cudaSuccess);
}

inline char const* error_name(DeviceError status) noexcept
{
    return cudaGetErrorName(status);
}

inline char const* error_string(DeviceError status) noexcept
{
    return cudaGetErrorString(status);
}

namespace internal
{

template <typename TL>
struct is_same_size_t;

template <typename T1, typename T2>
struct is_same_size_t<meta::TL<T1, T2>>
    : std::bool_constant<sizeof(T1) == sizeof(T2)>
{};

template <typename TL>
using is_same_size = meta::Force<is_same_size_t<TL>>;

} // namespace internal

template <typename... KernelArgs, typename... Args>
void launch_kernel_internal(void (*kernel)(KernelArgs...),
                            const dim3& grid_dim,
                            const dim3& block_dim,
                            std::size_t shared_mem,
                            DeviceStream stream,
                            Args&&... args)
{
  // Assumes KernelArgs and Args have been checked to be convertible.
  // If each Args type is the same size as its corresponding KernelArgs
  // type we can directly take pointers. Otherwise we need to do an
  // explicit conversion because sizes will be wrong.
  if constexpr (meta::tlist::FoldlTL<
                    meta::And,
                    std::bool_constant<true>,
                    meta::tlist::MapTL<
                        internal::is_same_size,
                        meta::tlist::ZipTL<meta::TL<KernelArgs...>,
                                           meta::TL<Args...>>>>::value)
  {
    void* kernel_args[] = {(void*) &args...};
    H2_CHECK_CUDA(cudaLaunchKernel((const void*) kernel,
                                   grid_dim,
                                   block_dim,
                                   kernel_args,
                                   shared_mem,
                                   stream));
  }
  else
  {
    auto converted_args =
        std::tuple<KernelArgs...>{std::forward<Args>(args)...};
    std::array<void*, sizeof...(Args)> kernel_args{
        std::apply([](auto&&... args_) { return ((void*) &args_, ...); })};
    H2_CHECK_CUDA(cudaLaunchKernel((const void*) kernel,
                                   grid_dim,
                                   block_dim,
                                   kernel_args.data(),
                                   shared_mem,
                                   stream));
  }
}

} // namespace gpu
} // namespace h2
