#pragma once

#include <distconv_config.hpp>

#if H2_HAS_CUDA
#include "memory_cuda.hpp"
#define TENSOR_CHECK_GPU(...) TENSOR_CHECK_CUDA(__VA_ARGS__)
#elif H2_HAS_ROCM
#include "memory_rocm.hpp"
#define TENSOR_CHECK_GPU(...) TENSOR_CHECK_HIP(__VA_ARGS__)
#endif
