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


namespace h2
{

// Define these here to avoid duplication.

inline std::ostream& operator<<(std::ostream& os,
                                const ComputeStream<Device::GPU>& stream)
{
  os << "GPU compute stream (" << ((void*) stream.get_stream()) << ")";
  return os;
}

inline std::ostream& operator<<(std::ostream& os,
                                const SyncEvent<Device::GPU>& event)
{
  os << "GPU sync event (" << ((void*) event.get_event()) << ")";
  return os;
}

}  // namespace h2
