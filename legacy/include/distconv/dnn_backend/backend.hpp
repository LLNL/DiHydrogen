#pragma once
#ifndef H2_LEGACY_INCLUDE_DISTCONV_BACKEND_HPP_INCLUDED
#define H2_LEGACY_INCLUDE_DISTCONV_BACKEND_HPP_INCLUDED

#include <distconv_config.hpp>

#include "dnn_backend.hpp"
#include "dnn_backend_impl.hpp"

// LBANN is looking for this in the distconv ns
namespace distconv
{
using BackendDNNLib = DNNBackend<distconv::GPUDNNBackend>;
// FIXME trb: HACK to make LBANN compile
namespace backend
{
using Options = ::distconv::Options;
} // namespace backend
} // namespace distconv

#if H2_HAS_CUDA

#define GPU_PROFILE_RANGE_POP nvtxRangePop
#define GPU_PROFILE_RANGE_PUSH nvtxRangePushA

#elif H2_HAS_ROCM

#define GPU_PROFILE_RANGE_POP roctxRangePop
#define GPU_PROFILE_RANGE_PUSH roctxRangePushA

#endif

#endif // H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
