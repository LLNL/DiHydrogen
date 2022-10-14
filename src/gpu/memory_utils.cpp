#include "h2/gpu/memory_utils.hpp"

#include "h2_config.hpp"

#if H2_HAS_CUDA
#include <cub/util_allocator.cuh>
#include <cuda_runtime.h>
#elif H2_HAS_ROCM
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
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

namespace
{

#define H2_GET_UINT_ENV_FUNC(FUNC_NAME, VAR, DEFAULT_VAL)                      \
    static unsigned int FUNC_NAME() noexcept                                   \
    {                                                                          \
        char const* env = std::getenv(VAR);                                    \
        return (env ? static_cast<unsigned int>(std::atoi(env))                \
                    : DEFAULT_VAL);                                            \
    }
H2_GET_UINT_ENV_FUNC(growth_factor, "H2_CUB_BIN_GROWTH", 2)
H2_GET_UINT_ENV_FUNC(min_bin, "H2_CUB_MIN_BIN", 1)
H2_GET_UINT_ENV_FUNC(max_bin,
                     "H2_CUB_MAX_BIN",
                     h2::gpu::RawCUBAllocType::INVALID_BIN)
#undef H2_GET_UINT_ENV_FUNC

static size_t max_cached_size() noexcept
{
    char const* env = std::getenv("H2_CUB_MAX_CACHED_SIZE");
    return (env ? static_cast<size_t>(std::atoll(env))
                : h2::gpu::RawCUBAllocType::INVALID_SIZE);
}

static bool debug() noexcept
{
    char const* env = std::getenv("H2_CUB_DEBUG");
    return (env && std::strlen(env) && env[0] != '0');
}

h2::gpu::RawCUBAllocType make_allocator(unsigned int const gf,
                                        unsigned int const min,
                                        unsigned int const max,
                                        size_t const max_cached,
                                        bool const debug)
{
    H2_GPU_INFO("hipcub allocator"
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
} // namespace

h2::gpu::RawCUBAllocType& h2::gpu::default_cub_allocator()
{
    static RawCUBAllocType alloc = make_allocator(
        growth_factor(), min_bin(), max_bin(), max_cached_size(), debug());
    return alloc;
}
