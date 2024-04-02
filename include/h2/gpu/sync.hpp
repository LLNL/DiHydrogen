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

/** Support printing GPU compute streams. */
inline std::ostream& operator<<(std::ostream& os,
                                const ComputeStream<Device::GPU>& stream)
{
  os << "GPU compute stream (" << ((void*) stream.get_stream()) << ")";
  return os;
}

/** Support printing GPU events. */
inline std::ostream& operator<<(std::ostream& os,
                                const SyncEvent<Device::GPU>& event)
{
  os << "GPU sync event (" << ((void*) event.get_event()) << ")";
  return os;
}

/** Equality for GPU events. */
inline bool operator==(const SyncEvent<Device::GPU>& event1,
                       const SyncEvent<Device::GPU>& event2) H2_NOEXCEPT
{
  return event1.get_event() == event2.get_event();
}

/** Inequality for GPU events. */
inline bool operator!=(const SyncEvent<Device::GPU>& event1,
                       const SyncEvent<Device::GPU>& event2) H2_NOEXCEPT
{
  return event1.get_event() != event2.get_event();
}

/** Equality for GPU streams. */
inline bool operator==(const ComputeStream<Device::GPU>& stream1,
                       const ComputeStream<Device::GPU>& stream2) H2_NOEXCEPT
{
  return stream1.get_stream() == stream2.get_stream();
}

/** Inequality for GPU streams. */
inline bool operator!=(const ComputeStream<Device::GPU>& stream1,
                       const ComputeStream<Device::GPU>& stream2) H2_NOEXCEPT
{
  return stream1.get_stream() != stream2.get_stream();
}

}  // namespace h2

namespace std
{

// Inject hash specializations for GPU sync objects.

template <>
struct hash<h2::SyncEvent<h2::Device::GPU>>
{
  size_t operator()(const h2::SyncEvent<h2::Device::GPU>& event) const noexcept
  {
    return hash<void*>()((void*) event.get_event());
  }
};

template <>
struct hash<h2::ComputeStream<h2::Device::GPU>>
{
  size_t
  operator()(const h2::ComputeStream<h2::Device::GPU>& stream) const noexcept
  {
    return hash<void*>()((void*) stream.get_stream());
  }
};

}
