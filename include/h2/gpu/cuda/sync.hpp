////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * CUDA implementation for GPU synchronization.
 */

#include "h2/core/sync.hpp"

#include <type_traits>

#include <El.hpp>

#include "h2/gpu/runtime.hpp"
#include "h2/gpu/logger.hpp"

#ifndef HYDROGEN_HAVE_ALUMINUM
#error "Aluminum support is required"
#endif

// We use Aluminum's internal event pool for simplicity.
#include <Al.hpp>
#ifndef AL_HAS_CUDA
#error "Aluminum GPU support is required"
#endif
static_assert(std::is_same_v<AlGpuEvent_t, cudaEvent_t>,
              "Aluminum GPU events are not CUDA events");

namespace h2
{

namespace internal
{

inline cudaEvent_t get_new_cuda_event()
{
  return Al::internal::cuda::event_pool.get();
}

inline void release_cuda_event(cudaEvent_t event)
{
  Al::internal::cuda::event_pool.release(event);
}

}  // namespace internal


template <>
class SyncEvent<Device::GPU> final : public SyncEventBase
{
public:
  static constexpr Device device = Device::GPU;

  /** Use a default, shared CUDA event. */
  SyncEvent()
  {
    // Copy Elemental's semantics here.
    event = El::cuda::GetDefaultEvent();
  }

  /** Wrap an existing CUDA event. */
  SyncEvent(cudaEvent_t event_) : event(event_) {}

  Device get_device() const override H2_NOEXCEPT { return Device::GPU; }

  void wait_for_this() const override
  {
    H2_GPU_TRACE("synchronizing event {}", (void*) event);
    H2_CHECK_CUDA(cudaEventSynchronize(event));
  }

  /** Return the underlying CUDA event. */
  cudaEvent_t get_event() const H2_NOEXCEPT { return event; }

private:
  cudaEvent_t event = nullptr;  /**< Encapsulated CUDA event. */

  friend void destroy_sync_event<Device::GPU>(SyncEvent<Device::GPU>&);
};

template <>
class ComputeStream<Device::GPU> final : public ComputeStreamBase
{
public:
  static constexpr Device device = Device::GPU;

  /** Use the default CUDA stream. */
  ComputeStream() : ComputeStream(El::cuda::GetDefaultStream()) {}

  /** Wrap an existing CUDA stream. */
  ComputeStream(cudaStream_t stream_) : stream(stream_) {}

  /** Support conversion from El::SyncInfo. */
  explicit ComputeStream(const El::SyncInfo<Device::GPU>& sync_info)
      : stream(sync_info.Stream())
  {}

  /** Support conversion to El::SyncInfo. */
  explicit operator El::SyncInfo<Device::GPU>() const H2_NOEXCEPT
  {
    return El::SyncInfo<Device::GPU>(stream, El::cuda::GetDefaultEvent());
  }

  Device get_device() const override H2_NOEXCEPT { return Device::GPU; }

  void add_sync_point(const SyncEventBase& event) const override
  {
    H2_ASSERT_DEBUG(event.get_device() == Device::GPU,
                    "SyncEvent for GPU stream must be a GPU type");
    auto gpu_event = static_cast<const SyncEvent<Device::GPU>&>(event);
    add_sync_point(gpu_event);
  }

  void add_sync_point(const SyncEvent<Device::GPU>& event) const
  {
    H2_GPU_TRACE("recording event {} on stream {}",
                 (void*) event.get_event(),
                 (void*) stream);
    H2_CHECK_CUDA(cudaEventRecord(event.get_event(), stream));
  }

  void wait_for(const SyncEventBase& event) const override
  {
    // If this is a CPU event, we do nothing.
    // If this is a GPU event, have this stream wait on it.
    if (event.get_device() == Device::GPU)
    {
      auto gpu_event = static_cast<const SyncEvent<Device::GPU>&>(event);
      wait_for(gpu_event);
    }
  }

  void wait_for(const SyncEvent<Device::CPU>&) const H2_NOEXCEPT {}

  void wait_for(const SyncEvent<Device::GPU>& event) const
  {
    H2_GPU_TRACE("stream {} waiting for event {}",
                 (void*) stream,
                 (void*) event.get_event());
    H2_CHECK_CUDA(cudaStreamWaitEvent(stream, event.get_event(), 0));
  }

  void wait_for(const ComputeStreamBase& other_stream) const override
  {
    // If this is a CPU stream, we do nothing (GPU streams do not wait
    // for CPU streams).
    // If this is a GPU stream different from this one, record an event
    // to capture all work on the stream and then wait for it.
    if (other_stream.get_device() == Device::GPU)
    {
      auto other_gpu_stream =
          static_cast<const ComputeStream<Device::GPU>&>(other_stream);
      wait_for(other_gpu_stream);
    }
  }

  void wait_for(const ComputeStream<Device::CPU>&) const H2_NOEXCEPT {}

  void wait_for(const ComputeStream<Device::GPU>& other_stream) const
  {
    if (stream == other_stream.stream)
    {
      return;
    }
    cudaEvent_t event = internal::get_new_cuda_event();
    H2_GPU_TRACE("recording event {} on stream {}",
                 (void*) event,
                 (void*) other_stream.stream);
    H2_CHECK_CUDA(cudaEventRecord(event, other_stream.stream));
    H2_GPU_TRACE(
        "stream {} waiting for event {}", (void*) stream, (void*) event);
    H2_CHECK_CUDA(cudaStreamWaitEvent(stream, event, 0));
    // Can release the event immediately, CUDA has already captured
    // the relevant work.
    internal::release_cuda_event(event);
  }

  void wait_for_this() const override
  {
    H2_GPU_TRACE("synchronizing stream {}", (void*) stream);
    H2_CHECK_CUDA(cudaStreamSynchronize(stream));
  }

  /** Return the underlying CUDA stream. */
  cudaStream_t get_stream() const H2_NOEXCEPT { return stream; }

private:
  cudaStream_t stream = nullptr; /**< Encapsulated CUDA stream. */

  friend void destroy_compute_stream<Device::GPU>(ComputeStream<Device::GPU>&);
};

template <>
inline ComputeStream<Device::GPU> create_new_compute_stream<Device::GPU>()
{
  return ComputeStream<Device::GPU>(gpu::make_stream());
}

template <>
inline void
destroy_compute_stream<Device::GPU>(ComputeStream<Device::GPU>& stream)
{
  gpu::destroy(stream.stream);
  stream.stream = nullptr;
}

template <>
inline SyncEvent<Device::GPU> create_new_sync_event<Device::GPU>()
{
  return SyncEvent<Device::GPU>(internal::get_new_cuda_event());
}

template <>
inline void destroy_sync_event<Device::GPU>(SyncEvent<Device::GPU>& event)
{
  internal::release_cuda_event(event.event);
  event.event = nullptr;
}

}
