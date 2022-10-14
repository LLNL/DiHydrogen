#pragma once

#include <distconv_config.hpp>
#if H2_HAS_CUDA
#include "runtime_cuda.hpp"
#elif H2_HAS_ROCM
#include "runtime_rocm.hpp"
#endif

namespace distconv
{
namespace tensor
{
namespace internal
{

#if H2_HAS_CUDA
using RuntimeGPU = RuntimeCUDA;
#elif H2_HAS_ROCM
using RuntimeGPU = RuntimeHIP;
#endif

} // namespace internal
} // namespace tensor
} // namespace distconv
