////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "h2/gpu/memory_utils.hpp"

#include "h2_config.hpp"

#include <hydrogen/device/gpu/CUB.hpp>

#if H2_HAS_CUDA
#include <cub/util_allocator.cuh>
#include <cuda_runtime.h>
#elif H2_HAS_ROCM
#include <hip/hip_runtime.h>
#include <hipcub/util_allocator.hpp>
#endif

#include <cstdlib>

// Note: The behavior of functions in this file may be impacted by the
// following user-provided environment variables (these corresponde
// directly to the ctor args for the memory pool):
//
//   - H2_CUB_BIN_GROWTH (uint32): The (geometric) growth factor for
//                                 the bin sizes in (HIP)CUB. Default:
//                                 2u.
//
//   - H2_CUB_MIN_BIN (uint32): Size in bytes of smallest bin in
//                              (HIP)CUB.  Default: 1u.
//
//   - H2_CUB_MAX_BIN (uint32): Size of the largest bin in
//                              (HIP)CUB. Default: no limit (leave
//                              unset).
//
//   - H2_CUB_MAX_CACHED_SIZE (uint64): Max aggregate cached bytes per
//                                      device in (HIP)CUB. Default:
//                                      no limit (leave unset).
//
//   - H2_CUB_DEBUG (bool): If true, print allocation/deallocation
//                          logs from (HIP)CUB to stdout. Default:
//                          false.
//
// As usual, boolean environment variables are truthy if they are set
// to any nonempty value that does not begin with '0'. That is, they
// match '[^0].*'. The behavior is undefined if the value of the H2_*
// variables differs across processes in one MPI universe.

unsigned int h2::gpu::cub_growth_factor() noexcept
{
    char const* env = std::getenv("H2_CUB_BIN_GROWTH");
    return (env ? static_cast<unsigned int>(std::atoi(env)) : 2);
}

unsigned int h2::gpu::cub_min_bin() noexcept
{
    char const* env = std::getenv("H2_CUB_MIN_BIN");
    return (env ? static_cast<unsigned int>(std::atoi(env)) : 1);
}

unsigned int h2::gpu::cub_max_bin() noexcept
{
    char const* env = std::getenv("H2_CUB_MAX_BIN");
    return (env ? static_cast<unsigned int>(std::atoi(env))
                : h2::gpu::RawCUBAllocType::INVALID_BIN);
}

size_t h2::gpu::cub_max_cached_size() noexcept
{
    char const* env = std::getenv("H2_CUB_MAX_CACHED_SIZE");
    return (env ? static_cast<size_t>(std::atoll(env))
                : h2::gpu::RawCUBAllocType::INVALID_SIZE);
}

bool h2::gpu::cub_debug() noexcept
{
    char const* env = std::getenv("H2_CUB_DEBUG");
    return (env && std::strlen(env) && env[0] != '0');
}

h2::gpu::RawCUBAllocType h2::gpu::make_allocator(unsigned int const gf,
                                                 unsigned int const min,
                                                 unsigned int const max,
                                                 size_t const max_cached,
                                                 bool const debug)
{
    H2_GPU_TRACE("H2 created CUB allocator"
                 "(gf={}, min={}, max={}, max_cached={}, debug={})",
                 gf,
                 min,
                 max,
                 max_cached,
                 debug);
    return h2::gpu::RawCUBAllocType{/*bin_growth=*/gf,
                                    /*min_bin=*/min,
                                    /*max_bin=*/max,
                                    /*max_cached_bytes=*/max_cached,
                                    /*skip_cleanup=*/false,
                                    /*debug=*/debug};
}

static bool use_internal_pool() noexcept
{
    char const* env = std::getenv("H2_INTERNAL_CUB_POOL");
    return (env && std::strlen(env) && env[0] != '0');
}

static h2::gpu::RawCUBAllocType& borrow_hydrogen_cub_allocator()
{
    auto& alloc = hydrogen::cub::MemoryPool();
    H2_GPU_TRACE("H2 using Hydrogen CUB allocator"
                 "(gf={}, min={}, max={}, max_cached={}, debug={})",
                 alloc.bin_growth,
                 alloc.min_bin,
                 alloc.max_bin,
                 alloc.max_cached_bytes,
                 alloc.debug);
    return alloc;
}

static h2::gpu::RawCUBAllocType& get_internal_cub_allocator()
{
    static auto alloc = h2::gpu::make_allocator();
    return alloc;
}

h2::gpu::RawCUBAllocType& h2::gpu::default_cub_allocator()
{
    static auto& alloc =
        (use_internal_pool() ? get_internal_cub_allocator()
                             : borrow_hydrogen_cub_allocator());
    return alloc;
}
