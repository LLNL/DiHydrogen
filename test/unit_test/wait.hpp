////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <h2_config.hpp>

#include "h2/core/sync.hpp"

#ifdef H2_HAS_GPU

#include "h2/gpu/runtime.hpp"

/**
 * Enqueue a kernel on stream that executes for approximately length
 * seconds before completing.
 *
 * This is intended to be used in tests (or similar) where some work on
 * a stream is needed to simulate real work.
 */
void gpu_wait(double length, h2::gpu::DeviceStream stream);

/** Variant that takes a ComputeStream instead. */
void gpu_wait(double length, const h2::ComputeStream& stream);

#endif  // H2_HAS_GPU
