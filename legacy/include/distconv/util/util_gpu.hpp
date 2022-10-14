#pragma once

#include "distconv_config.hpp"

#if H2_HAS_CUDA

#include "./util_cuda.hpp"
#define DISTCONV_CHECK_GPU(...) DISTCONV_CHECK_CUDA(__VA_ARGS__)
#define DISTCONV_GPU_MALLOC(...) DISTCONV_CUDA_MALLOC(__VA_ARGS__)

#elif H2_HAS_ROCM

#include "./util_rocm.hpp"
#define DISTCONV_CHECK_GPU(...) DISTCONV_CHECK_HIP(__VA_ARGS__)
#define DISTCONV_GPU_MALLOC(...) DISTCONV_HIP_MALLOC(__VA_ARGS__)

#endif
