#pragma once

#include <distconv_config.hpp>

#if H2_HAS_CUDA
#include "stream_cuda.hpp"
#elif H2_HAS_ROCM
#include "stream_rocm.hpp"
#endif
