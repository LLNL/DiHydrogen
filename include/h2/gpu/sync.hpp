////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * GPU synchronization primitives.
 */

#include <h2_config.hpp>

#if H2_HAS_CUDA
#include "cuda/sync.hpp"
#elif H2_HAS_ROCM
#include "rocm/sync.hpp"
#endif
