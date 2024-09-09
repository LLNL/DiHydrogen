////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <distconv_config.hpp>

#include "dnn_backend.hpp"
#include "dnn_backend_impl.hpp"

#if H2_HAS_DACE
#include "dace_backend.hpp"
#endif

// LBANN is looking for this in the distconv ns
namespace distconv
{
#if H2_HAS_DACE
using BackendDNNLib = DaCeDNNBackend<distconv::GPUDNNBackend>;
#else
using BackendDNNLib = DNNBackend<distconv::GPUDNNBackend>;
#endif
// FIXME trb: HACK to make LBANN compile
namespace backend
{
using Options = ::distconv::Options;
}  // namespace backend
}  // namespace distconv

#if H2_HAS_CUDA

#define GPU_PROFILE_RANGE_POP nvtxRangePop
#define GPU_PROFILE_RANGE_PUSH nvtxRangePushA

#elif H2_HAS_ROCM

#define GPU_PROFILE_RANGE_POP roctxRangePop
#define GPU_PROFILE_RANGE_PUSH roctxRangePushA

#endif
