#pragma once
#ifndef H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
#define H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED

#include <distconv_config.hpp>

#if H2_HAS_CUDA
#include "backend_cudnn.hpp"
namespace distconv
{
using BackendDNNLib = cudnn::BackendCUDNN;
namespace backend = cudnn;
} // namespace distconv

#define GPU_PROFILE_RANGE_POP nvtxRangePop
#define GPU_PROFILE_RANGE_PUSH nvtxRangePushA

#elif H2_HAS_ROCM
#include "backend_miopen.hpp"
namespace distconv
{
using BackendDNNLib = miopen::BackendMIOpen;
namespace backend = miopen;
} // namespace distconv

#define GPU_PROFILE_RANGE_POP roctxRangePop
#define GPU_PROFILE_RANGE_PUSH roctxRangePushA

#endif

#include "distconv/cudnn/batchnorm.hpp"
#include "distconv/cudnn/convolution.hpp"
#include "distconv/cudnn/cross_entropy.hpp"
#include "distconv/cudnn/leaky_relu.hpp"
#include "distconv/cudnn/mean_squared_error.hpp"
#include "distconv/cudnn/pooling.hpp"
#include "distconv/cudnn/relu.hpp"
#include "distconv/cudnn/softmax.hpp"

#endif // H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
