 ////////////////////////////////////////////////////////////////////////////////
 // Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
 // DiHydrogen Project Developers. See the top-level LICENSE file for details.
 //
 // SPDX-License-Identifier: Apache-2.0
 ////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>

#include "h2/tensor/tensor_types.hpp"
#include "h2/meta/TypeList.hpp"

// List of device types that will be tested by Catch2.
// The raw CPU/GPUDev_t can be used in TEMPLATE_TEST_CASE and the
// AllDevList can be used in TEMPLATE_LIST_TEST_CASE.
using CPUDev_t = std::integral_constant<h2::Device, h2::Device::CPU>;
#ifdef H2_TEST_WITH_GPU
using GPUDev_t = std::integral_constant<h2::Device, h2::Device::GPU>;
using AllDevList = h2::meta::TL<CPUDev_t, GPUDev_t>;
#else
using AllDevList = h2::meta::TL<CPUDev_t>;
#endif

// Standard datatype to be used when testing.
// Note: When used with integers, floats are exact for any integer less
// than 2^24.
using DataType = float;
