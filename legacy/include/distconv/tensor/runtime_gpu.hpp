////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
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
