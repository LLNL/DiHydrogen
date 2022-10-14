#pragma once
#ifndef H2_INCLUDE_H2_GPU_RUNTIME_HPP_INCLUDED
#define H2_INCLUDE_H2_GPU_RUNTIME_HPP_INCLUDED

/** @file
 *
 *  The public runtime API that is exposed for user and library
 *  consumption. These are lightweight wrappers around HIP or CUDA
 *  runtime functions. The are accessible in the h2::gpu namespace.
 *
 *  typedef {cuda,hip}Stream_t DeviceStream;
 *  typedef {cuda,hip}Event_t DeviceEvent;
 *
 *  int num_gpus();
 *  int current_gpu();
 *  void set_gpu(int id);
 *
 *  void init_runtime();
 *  void finalize_runtime();
 *  bool runtime_is_initialized();
 *  bool runtime_is_finalized();
 *
 *  bool ok(DeviceError) noexcept;
 *
 *  DeviceStream make_stream();
 *  DeviceStream make_stream_nonblocking();
 *  void destroy(DeviceStream);
 *
 *  DeviceEvent make_event();
 *  DeviceEvent make_event_notiming();
 *  void destroy(DeviceEvent);
 *
 *  void sync();             // Device Sync
 *  void sync(DeviceEvent);  // Sync on event.
 *  void sync(DeviceStream); // Sync on stream.
 */

#include "h2_config.hpp"

// This adds the runtime-specific stuff.
#if H2_HAS_CUDA
#include "cuda/runtime.hpp"
#elif H2_HAS_ROCM
#include "rocm/runtime.hpp"
#endif

// This declares the rest of the general API.
namespace h2
{
namespace gpu
{

int num_gpus();
int current_gpu();
void set_gpu(int id);

void init_runtime();
void finalize_runtime();
bool runtime_is_initialized();
bool runtime_is_finalized();

DeviceStream make_stream();
DeviceStream make_stream_nonblocking();
void destroy(DeviceStream);

DeviceEvent make_event();
DeviceEvent make_event_notiming();
void destroy(DeviceEvent);

void sync();             // Device Sync
void sync(DeviceEvent);  // Sync on event.
void sync(DeviceStream); // Sync on stream.

} // namespace gpu
} // namespace h2

#endif // H2_INCLUDE_H2_GPU_RUNTIME_HPP_INCLUDED
