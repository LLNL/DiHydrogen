#pragma once
#ifndef H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
#define H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED

#include <distconv_config.hpp>

#if H2_HAS_CUDA
#include "backend_cudnn.hpp"
namespace distconv
{
using BackendDNNLib = cudnn::BackendCUDNN;
namespace dnn_lib = cudnn;
namespace backend = dnn_lib;
} // namespace distconv

#define GPU_PROFILE_RANGE_POP nvtxRangePop
#define GPU_PROFILE_RANGE_PUSH nvtxRangePushA

#elif H2_HAS_ROCM
#include "backend_miopen.hpp"
namespace distconv
{
using BackendDNNLib = miopen::BackendMIOpen;
namespace dnn_lib = miopen;
namespace backend = dnn_lib;
} // namespace distconv

#define GPU_PROFILE_RANGE_POP roctxRangePop
#define GPU_PROFILE_RANGE_PUSH roctxRangePushA

#endif

#include "distconv/dnn_backend/batchnorm.hpp"
#include "distconv/dnn_backend/convolution.hpp"
#include "distconv/dnn_backend/cross_entropy.hpp"
#include "distconv/dnn_backend/leaky_relu.hpp"
#include "distconv/dnn_backend/mean_squared_error.hpp"
#include "distconv/dnn_backend/pooling.hpp"
#include "distconv/dnn_backend/relu.hpp"
#include "distconv/dnn_backend/softmax.hpp"

#endif // H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
