////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once
#ifndef H2_LEGACY_INCLUDE_DISTCONV_UTIL_UTIL_GPU_DNN_HPP_INCLUDED
#define H2_LEGACY_INCLUDE_DISTCONV_UTIL_UTIL_GPU_DNN_HPP_INCLUDED

#include <distconv_config.hpp>
#if H2_HAS_CUDA
#include "./util_cudnn.hpp"
#elif H2_HAS_ROCM
#include "./util_miopen.hpp"
#endif

#endif // H2_LEGACY_INCLUDE_DISTCONV_UTIL_UTIL_GPU_DNN_HPP_INCLUDED
