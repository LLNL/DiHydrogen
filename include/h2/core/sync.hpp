////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Synchronization primitives.
 */

#include <h2_config.hpp>

#include <El.hpp>

#include <tuple>
#include <utility>

#include "h2/core/device.hpp"
#include "h2/utils/Error.hpp"

#ifdef H2_HAS_GPU
#include "h2/gpu/runtime.hpp"

// We use Aluminum's internal event pool for GPU events.
#ifndef HYDROGEN_HAVE_ALUMINUM
#error "Aluminum support is required"
#endif
#include <Al.hpp>

#endif

namespace h2
{

/**
 * Manage device-specific synchronization.
 *
 * A note on how Tensors deal with synchronization (here because there
 * isn't a great place to write this since it touches many classes):
 *
 * When a Tensor is created, the creator may either specify the
 * ComputeStream for the Tensor to use, or the Tensor will create one
 * with the default ComputeStream constructor for the appropriate
 * Device. This ComputeStream is passed through to the underlying
 * StridedMemory and RawBuffer. The RawBuffer will allocate any memory
 * using that ComputeStream. Any Tensor operation that changes the
 * underlying RawBuffer (e.g., `empty`, `resize`) will continue to use
 * the ComputeStream associated with the Tensor. As a special case of
 * this, an empty Tensor, which has no RawBuffer, will use the Tensor's
 * ComputeStream should it construct a RawBuffer (e.g., due to being
 * resized).
 *
 * When a view of a Tensor is created, the viewing Tensor will default
 * to the same ComputeStream as the original Tensor.
 *
 * When a Tensor wraps external memory (by providing a raw pointer),
 * there is again no RawBuffer created and the Tensor's ComputeStream
 * will be used for all operations.
 *
 * The get/set_stream methods may be used on Tensors and RawBuffers
 * to retrieve or change the associated ComputeStream. get_stream on a
 * Tensor always returns the Tensor's ComputeStream, which may be
 * different from the ComputeStream  associated with the RawBuffer
 * underlying the Tensor (due to set_stream).
 *
 * If the ComputeStream on a Tensor is changed (via set_stream), the
 * semantics depend on whether the Tensor is a view. If the Tensor is
 * not a view, this will also change the stream of the underlying
 * RawBuffer. If the Tensor is a view, only the Tensor's stream will
 * be changed. (This is how a Tensor's stream may differ from its
 * RawBuffer's.) This enables views of the same Tensor to enqueue
 * operations on multiple compute streams concurrently; it is up to the
 * user to ensure the appropriate synchronization in such uses.
 *
 * This requires careful handling of destruction in the RawBuffer, as
 * there may be operations on multiple compute streams accessing the
 * data, yet the RawBuffer is only (directly) associated with one
 * ComputeStream. In particular, consider the case where an initial
 * Tensor A is created with ComputeStream SA, and then a view, B, of
 * that Tensor is created and associated with ComputeStream SB. The
 * underlying RawBuffer will be associated with SA. If A is deleted,
 * the RawBuffer will still exist, as B still has a reference to it.
 * Now suppose B launches some operations on SB, then is itself
 * deleted. The operations should continue to run fine, due to the
 * stream ordering. However, the RawBuffer's deletion will be
 * synchronized only to SA, potentially leading to a race with
 * operations on SB. To avoid this, whenever a Tensor discards a
 * reference to a RawBuffer, it informs the RawBuffer it is doing so,
 * along with its current ComputeStream. If the stream differs from the
 * RawBuffer's, it will record an event on the Tensor's ComputeStream
 *  and keep a reference to it. When the RawBuffer is deleted, it will
 * synchronize with all recorded events before enqueuing the delete, to
 * avoid races.
 *
 * Another situation to be aware of: If you change a Tensor's
 * ComputeStream, it is up to you to provide any needed synchronization
 * between the original ComputeStream and the new one.
 *
 * An implementation note (separate from the above semantics):
 * ComputeStream objects are stored in StridedMemory, rather than
 * directly in a Tensor. This is just to simplify implementation.
 */

namespace internal
{

/** Define the underlying event type used by a device. */
template <Device Dev>
struct RawSyncEvent {};

template <>
struct RawSyncEvent<Device::CPU>
{
  using type = int;
};

#ifdef H2_HAS_GPU
template <>
struct RawSyncEvent<Device::GPU>
{
  using type = gpu::DeviceEvent;
};

static_assert(std::is_same_v<AlGpuEvent_t, gpu::DeviceEvent>,
              "Aluminum GPU events are not DeviceEvents");

// Wrap Aluminum's internal event pool.
inline gpu::DeviceEvent get_new_device_event()
{
  return Al::internal::cuda::event_pool.get();
}

inline void release_device_event(gpu::DeviceEvent event)
{
  Al::internal::cuda::event_pool.release(event);
}

#endif  // H2_HAS_GPU

/** Define the underlying compute stream type used by a device. */
template <Device Dev>
struct RawComputeStream {};

template <>
struct RawComputeStream<Device::CPU>
{
  using type = int;
};

#ifdef H2_HAS_GPU
template <>
struct RawComputeStream<Device::GPU>
{
  using type = gpu::DeviceStream;
};
#endif

/** Get the default event for a device. */
template <Device Dev>
inline typename RawSyncEvent<Dev>::type get_default_event();

template <>
inline typename RawSyncEvent<Device::CPU>::type get_default_event<Device::CPU>()
{
  return 0;
}

#ifdef H2_HAS_GPU
template <>
inline typename RawSyncEvent<Device::GPU>::type get_default_event<Device::GPU>()
{
#if H2_HAS_CUDA
  return El::cuda::GetDefaultEvent();
#elif H2_HAS_ROCM
  return El::rocm::GetDefaultEvent();
#endif
}
#endif

/** Get the default compute stream for a device. */
template <Device Dev>
inline typename RawComputeStream<Dev>::type get_default_compute_stream();

template <>
inline typename RawComputeStream<Device::CPU>::type get_default_compute_stream<Device::CPU>()
{
  return 0;
}

#ifdef H2_HAS_GPU
template <>
inline typename RawComputeStream<Device::GPU>::type get_default_compute_stream<Device::GPU>()
{
#if H2_HAS_CUDA
  return El::cuda::GetDefaultStream();
#elif H2_HAS_ROCM
  return El::rocm::GetDefaultStream();
#endif
}
#endif

}  // namespace internal

// Forward-declarations:
class SyncEvent;
class ComputeStream;
template <Device Dev>
inline void destroy_sync_event(SyncEvent&);
template <Device Dev>
inline void destroy_compute_stream(ComputeStream&);

/**
 * Device-specific synchronization objects.
 *
 * These are used to synchronize between ComputeStreams.
 *
 * On CPUs, these do nothing. On GPUs, these correspond to events.
 */
class SyncEvent
{
public:

  /** Create a new event using the device's default event. */
  SyncEvent(Device device_)
    : device(device_)
  {
    H2_DEVICE_DISPATCH(
      device,
      cpu_event = internal::get_default_event<Dev>(),
      gpu_event = internal::get_default_event<Dev>());
  }

#ifdef H2_HAS_GPU
  /** Wrap an existing GPU event. */
  SyncEvent(typename internal::RawSyncEvent<Device::GPU>::type raw_event)
    : device(Device::GPU), gpu_event(raw_event) {}
#endif

  SyncEvent(const SyncEvent&) = default;
  SyncEvent& operator=(const SyncEvent&) = default;

  SyncEvent(SyncEvent&& other)
    : device(other.device)
  {
    H2_DEVICE_DISPATCH(
        device,
        {
          cpu_event = std::exchange(other.cpu_event, 0);
        },
        {
          gpu_event = std::exchange(other.gpu_event, nullptr);
        });
  }
  SyncEvent& operator=(SyncEvent&& other)
  {
    device = other.device;
    H2_DEVICE_DISPATCH(
        device,
        {
          cpu_event = std::exchange(other.cpu_event, 0);
        },
        {
          gpu_event = std::exchange(other.gpu_event, nullptr);
        });
    return *this;
  }

  /** Return the device type of the event. */
  Device get_device() const H2_NOEXCEPT { return device; }

  /**
   * Wait for all work currently recorded by the event to complete.
   *
   * This does nothing if this is a CPU event. If this is a GPU event,
   * the caller will wait for it.
   */
  void wait_for_this() const
  {
    H2_DEVICE_DISPATCH_SAME(device, wait_for_this<Dev>());
  }

  template <Device Dev>
  void wait_for_this() const
  {
    H2_ASSERT_DEBUG(
        Dev == device, "Incorrect device ", Dev, " (expected ", device, ")");
    H2_DEVICE_DISPATCH_CONST(
      Dev,
      (void) 0,
      gpu::sync(gpu_event));
  }

  /** Return the underlying raw event for the device. */
  template <Device Dev>
  typename internal::RawSyncEvent<Dev>::type get_event() const H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(
        Dev == device, "Incorrect device ", Dev, " (expected ", device, ")");
    H2_DEVICE_DISPATCH_CONST(
      Dev,
      return cpu_event,
      return gpu_event);
  }

private:
  Device device;  /**< The device this event is for. */

  /** Holds the actual event type. */
  union
  {
    typename internal::RawSyncEvent<Device::CPU>::type cpu_event;
#ifdef H2_HAS_GPU
    typename internal::RawSyncEvent<Device::GPU>::type gpu_event;
#endif
  };

#ifdef H2_HAS_GPU
  template <Device D>
  friend void destroy_sync_event(SyncEvent&);
#endif
};

/** Support printing synchronization events. */
inline std::ostream& operator<<(std::ostream& os, const SyncEvent& event)
{
  H2_DEVICE_DISPATCH_SAME(
    event.get_device(),
    os << event.get_device() << " sync event ("
    << event.get_event<Dev>() << ")");
  return os;
}

/** Equality for synchronization events. */
inline bool operator==(const SyncEvent& event1, const SyncEvent& event2) H2_NOEXCEPT
{
  if (event1.get_device() != event2.get_device())
  {
    return false;
  }
  H2_DEVICE_DISPATCH(
    event1.get_device(),
    return true,
    return event1.get_event<Dev>() == event2.get_event<Dev>());
}

/** Inequality for synchronization events. */
inline bool operator!=(const SyncEvent& event1, const SyncEvent& event2) H2_NOEXCEPT
{
  if (event1.get_device() != event2.get_device())
  {
    return true;
  }
  H2_DEVICE_DISPATCH(
    event1.get_device(),
    return false,
    return event1.get_event<Dev>() != event2.get_event<Dev>());
}

/** Create a fresh synchronization event for a particular device. */
template <Device Dev>
inline SyncEvent create_new_sync_event()
{
  H2_DEVICE_DISPATCH_CONST(
    Dev,
    return SyncEvent{Dev},
    return SyncEvent{internal::get_new_device_event()});
}

inline SyncEvent create_new_sync_event(Device device)
{
  H2_DEVICE_DISPATCH_SAME(device, return create_new_sync_event<Dev>());
}

/** Destroy a synchronization event for a particular device. */
template <Device Dev>
inline void destroy_sync_event(SyncEvent& event)
{
  H2_DEVICE_DISPATCH_CONST(
    Dev,
    (void) event,
    if (event.gpu_event != nullptr) {
      internal::release_device_event(event.gpu_event);
      event.gpu_event = nullptr;
    });
}

inline void destroy_sync_event(SyncEvent& event)
{
  H2_DEVICE_DISPATCH_SAME(event.get_device(), destroy_sync_event<Dev>(event));
}

/**
 * RAII manager for events.
 *
 * If no device is provided, this defaults to a CPU event.
 */
class SyncEventRAII
{
public:

  SyncEventRAII() : event(create_new_sync_event<Device::CPU>()) {}

  SyncEventRAII(Device device) : event(create_new_sync_event(device)) {}

  ~SyncEventRAII() { destroy_sync_event(event); }

  // Prevent copying.
  SyncEventRAII(const SyncEventRAII&) = delete;
  SyncEventRAII& operator=(const SyncEventRAII&) = delete;

  // Move constructor/assignment needs to be provided explicitly.
  SyncEventRAII(SyncEventRAII&& other) : event(std::move(other.event)) {}
  SyncEventRAII& operator=(SyncEventRAII&& other)
  {
    event = std::move(other.event);
    return *this;
  }

  /** Allow conversion to a regular SyncEvent. */
  operator SyncEvent() const H2_NOEXCEPT { return event; }

  SyncEvent event;
};

/**
 * A device-specific compute stream.
 *
 * Compute streams represent an ordered sequence of computation, and
 * are associated with a particular device type. Computation may be
 * ordered between streams (including streams for different devices).
 * The underlying, device-specific stream may be obtained for actual
 * use by device-specific code. These are wrappers and do not directly
 * manage resources.
 *
 * On CPUs, these are empty, as we currently assume a single-threaded,
 * synchronous model. On GPUs, these correspond to streams.
 *
 * Currently, we operate with the following semantics:
 * - CPU streams are inherently ordered, so there is no synchronization
 * between them.
 * - A GPU stream is inherently ordered with respect to itself and so
 * there is no synchronization needed.
 * - Two different GPU streams can be synchronized (i.e., one waits on
 * the other).
 * - A CPU stream can be synchronized to a GPU stream (i.e., the CPU
 * waits on the GPU).
 * - A GPU stream never waits for a CPU stream. (This may be revised in
 * the future, but we currently do not need it.)
 *
 * Compute streams may be constructed from their corresponding Hydrogen
 * SyncInfo object; the event will be discrded. Likewise, they may be
 * converted to their corresponding Hydrogen SyncInfo objects, which
 * will use the default Hydrogen event.
 */
class ComputeStream
{
public:

  /** Create a new compute stream with the device's default stream. */
  ComputeStream(Device device_)
    : device(device_)
  {
    H2_DEVICE_DISPATCH(
      device,
      cpu_stream = internal::get_default_compute_stream<Dev>(),
      gpu_stream = internal::get_default_compute_stream<Dev>());
  }

#ifdef H2_HAS_GPU
  /** Wrap an existing device stream. */
  ComputeStream(typename internal::RawComputeStream<Device::GPU>::type raw_stream)
    : device(Device::GPU), gpu_stream(raw_stream) {}
#endif

  /** Support conversion from an existing El::SyncInfo. */
  template <Device Dev>
  explicit ComputeStream(const El::SyncInfo<Dev>& sync_info)
    : device(Dev)
  {
    H2_DEVICE_DISPATCH_CONST(
      Dev,
      cpu_stream = internal::get_default_compute_stream<Dev>(),
      gpu_stream = sync_info.Stream());
  }

  /** Support conversion to El::SyncInfo. */
  template <Device Dev>
  explicit operator El::SyncInfo<Dev>() const H2_NOEXCEPT
  {
    H2_DEVICE_DISPATCH_CONST(
      Dev,
      return El::SyncInfo<Dev>{},
      return El::SyncInfo<Dev>(get_stream<Dev>(),
                               internal::get_default_event<Dev>()));
  }

  ComputeStream(const ComputeStream&) = default;
  ComputeStream& operator=(const ComputeStream&) = default;

  ComputeStream(ComputeStream&& other) : device(other.device)
  {
    H2_DEVICE_DISPATCH(
        device,
        {
          cpu_stream = std::exchange(other.cpu_stream, 0);
        },
        {
          gpu_stream = std::exchange(other.gpu_stream, nullptr);
        });
  }
  ComputeStream& operator=(ComputeStream&& other)
  {
    device = other.device;
    H2_DEVICE_DISPATCH(
        device,
        {
          cpu_stream = std::exchange(other.cpu_stream, 0);
        },
        {
          gpu_stream = std::exchange(other.gpu_stream, nullptr);
        });
    return *this;
  }

  /** Return the device type of the stream. */
  Device get_device() const H2_NOEXCEPT { return device; }

  /** Record the current state of the stream in the given event. */
  void add_sync_point(const SyncEvent& event) const
  {
    H2_DEVICE_DISPATCH_SAME(device, add_sync_point<Dev>(event));
  }

  template <Device ThisDev>
  void add_sync_point(const SyncEvent& event) const
  {
    H2_DEVICE_DISPATCH_SAME(
      event.get_device(),
      (add_sync_point<ThisDev, Dev>(event)));
  }

  template <Device ThisDev, Device EventDev>
  void add_sync_point(const SyncEvent& event) const
  {
    H2_ASSERT_DEBUG(ThisDev == device,
                    "Incorrect device ",
                    ThisDev,
                    " (expected ",
                    device,
                    ")");
    if constexpr (ThisDev == Device::CPU)
    {
      if constexpr (EventDev == Device::CPU)
      {
        // Nothing to do: CPU is inherently ordered.
      }
#ifdef H2_HAS_GPU
      else if constexpr (EventDev == Device::GPU)
      {
        throw H2Exception("Cannot have CPU sync to GPU event");
      }
#endif
    }
#ifdef H2_HAS_GPU
    else if constexpr (ThisDev == Device::GPU)
    {
      if constexpr (EventDev == Device::CPU)
      {
        throw H2Exception("Cannot have GPU sync to CPU event");
      }
      else if constexpr (EventDev == Device::GPU)
      {
        gpu::record_event(event.get_event<Device::GPU>(), gpu_stream);
      }
    }
#endif
  }

  /** Have this stream wait for the provided event to complete. */
  void wait_for(const SyncEvent& event) const
  {
    H2_DEVICE_DISPATCH_SAME(device, wait_for<Dev>(event));
  }

  template <Device ThisDev>
  void wait_for(const SyncEvent& event) const
  {
    H2_DEVICE_DISPATCH_SAME(
      event.get_device(),
      (wait_for<ThisDev, Dev>(event)));
  }

  template <Device ThisDev, Device EventDev>
  void wait_for(const SyncEvent& event) const
  {
    H2_ASSERT_DEBUG(ThisDev == device,
                    "Incorrect device ",
                    ThisDev,
                    " (expected ",
                    device,
                    ")");
    if constexpr (ThisDev == Device::CPU)
    {
      if constexpr (EventDev == Device::CPU)
      {
        // Nothing to do: CPU is inherently ordered.
      }
#ifdef H2_HAS_GPU
      else if constexpr (EventDev == Device::GPU)
      {
        // CPU waits on the event.
        event.wait_for_this();
      }
#endif
    }
#ifdef H2_HAS_GPU
    else if constexpr (ThisDev == Device::GPU)
    {
      if constexpr (EventDev == Device::CPU)
      {
        // Nothing to do: GPUs do not wait on CPUs.
      }
      else if constexpr (EventDev == Device::GPU)
      {
        // This stream waits on the event.
        gpu::sync(gpu_stream, event.get_event<Device::GPU>());
      }
    }
#endif
  }

  /**
   * Have this stream wait for all work currently on the given stream
   * to complete.
   */
  void wait_for(const ComputeStream& other_stream) const
  {
    H2_DEVICE_DISPATCH_SAME(device, wait_for<Dev>(other_stream));
  }

  template <Device ThisDev>
  void wait_for(const ComputeStream& other_stream) const
  {
    H2_DEVICE_DISPATCH_SAME(
      other_stream.get_device(),
      (wait_for<ThisDev, Dev>(other_stream)));
  }

  template <Device ThisDev, Device StreamDev>
  void wait_for(const ComputeStream& other_stream) const
  {
    H2_ASSERT_DEBUG(ThisDev == device,
                    "Incorrect device ",
                    ThisDev,
                    " (expected ",
                    device,
                    ")");
    if constexpr (ThisDev == Device::CPU)
    {
      if constexpr (StreamDev == Device::CPU)
      {
        // Nothing to do: CPU is inherently ordered.
      }
#ifdef H2_HAS_GPU
      else if constexpr (StreamDev == Device::GPU)
      {
        // CPU waits on the stream.
        other_stream.wait_for_this();
      }
#endif
    }
#ifdef H2_HAS_GPU
    else if constexpr (ThisDev == Device::GPU)
    {
      if constexpr (StreamDev == Device::CPU)
      {
        // Nothing to do: GPUs do not wait on CPUs.
      }
      else if constexpr (StreamDev == Device::GPU)
      {
        // Add an event and wait on it.
        gpu::DeviceEvent event = internal::get_new_device_event();
        gpu::record_event(event, other_stream.get_stream<Device::GPU>());
        gpu::sync(gpu_stream, event);
        internal::release_device_event(event);
      }
    }
#endif
  }

  /**
   * Wait for all work currently on this stream to complete.
   *
   * This does nothing if this is a CPU stream. If this is a GPU
   * stream, the caller will wait for it.
   */
  void wait_for_this() const
  {
    H2_DEVICE_DISPATCH_SAME(device, wait_for_this<Dev>());
  }

  template <Device ThisDev>
  void wait_for_this() const
  {
    H2_ASSERT_DEBUG(ThisDev == device,
                    "Incorrect device ",
                    ThisDev,
                    " (expected ",
                    device,
                    ")");
    H2_DEVICE_DISPATCH_CONST(
      ThisDev,
      (void) 0,
      gpu::sync(gpu_stream));
  }

  /** Return the underlying raw stream for the device. */
  template <Device ThisDev>
  typename internal::RawComputeStream<ThisDev>::type get_stream() const H2_NOEXCEPT
  {
    H2_ASSERT_DEBUG(ThisDev == device,
                    "Attempt to get raw stream for wrong device ",
                    ThisDev,
                    " (expected ",
                    device,
                    ")");
    H2_DEVICE_DISPATCH_CONST(
      ThisDev,
      return cpu_stream,
      return gpu_stream);
  }

private:
  Device device;  /**< The device this stream is for. */

  /** Holds the actual stream type. */
  union
  {
    typename internal::RawComputeStream<Device::CPU>::type cpu_stream;
#ifdef H2_HAS_GPU
    typename internal::RawComputeStream<Device::GPU>::type gpu_stream;
#endif
  };

#ifdef H2_HAS_GPU
  template <Device D>
  friend void destroy_compute_stream(ComputeStream&);
#endif
};

/** Support printing compute streams. */
inline std::ostream& operator<<(std::ostream& os,
                                const ComputeStream& stream) H2_NOEXCEPT
{
  H2_DEVICE_DISPATCH_SAME(
    stream.get_device(),
    os << stream.get_device() << " stream ("
    << stream.get_stream<Dev>() << ")");
  return os;
}

/** Equality for compute streams. */
inline bool operator==(const ComputeStream& stream1, const ComputeStream& stream2) H2_NOEXCEPT
{
  if (stream1.get_device() != stream2.get_device())
  {
    return false;
  }
  H2_DEVICE_DISPATCH(
    stream1.get_device(),
    return true,
    return stream1.get_stream<Dev>() == stream2.get_stream<Dev>());
}

/** Inequality for compute streams. */
inline bool operator!=(const ComputeStream& stream1, const ComputeStream& stream2) H2_NOEXCEPT
{
  if (stream1.get_device() != stream2.get_device())
  {
    return true;
  }
  H2_DEVICE_DISPATCH(
    stream1.get_device(),
    return false,
    return stream1.get_stream<Dev>() != stream2.get_stream<Dev>());
}

/** Create a fresh compute stream for a particular device. */
template <Device Dev>
inline ComputeStream create_new_compute_stream()
{
  H2_DEVICE_DISPATCH_CONST(
    Dev,
    return ComputeStream{Dev},
    return ComputeStream{gpu::make_stream()});
}

inline ComputeStream create_new_compute_stream(Device device)
{
  H2_DEVICE_DISPATCH_SAME(device, return create_new_compute_stream<Dev>());
}

/** Destroy a compute stream for a particular device. */
template <Device Dev>
inline void destroy_compute_stream(ComputeStream& stream)
{
  H2_DEVICE_DISPATCH_CONST(
    Dev,
    (void) stream,
    if (stream.gpu_stream != nullptr) {
      gpu::destroy(stream.gpu_stream);
      stream.gpu_stream = nullptr;
    });
}

inline void destroy_compute_stream(ComputeStream& stream)
{
  H2_DEVICE_DISPATCH_SAME(
    stream.get_device(), destroy_compute_stream<Dev>(stream));
}

// General utilities for interacting with compute streams and events:

/**
 * Have the main stream wait for all work currently on the other
 * streams to complete.
 */
template <typename... OtherStreams>
inline void stream_wait_on_all(const ComputeStream& main,
                               const OtherStreams&... others)
{
  (main.wait_for(others), ...);
}

/**
 * Have the other streams wait for all work currently on the main
 * stream to complete.
 */
template <typename... OtherStreams>
inline void all_wait_on_stream(const ComputeStream& main,
                               const OtherStreams&... others)
{
  (others.wait_for(main), ...);
}

/**
 * Provide a RAII wrapper for synchronizing multiple compute streams.
 *
 * The first provided stream will be the primary stream. On creation,
 * the primary stream will wait for all other provided streams. On
 * destruction, all other streams will wait for the primary stream.
 *
 * This enables a common pattern where an operation may interact with
 * multiple different streams, but computation occurs on only one of
 * the streams.
 */
template <typename... OtherStreams>
class MultiSync
{
public:
  MultiSync(const ComputeStream& main_stream_,
            const OtherStreams&... other_streams_)
    : main_stream(main_stream_), other_streams(other_streams_...)
  {
    stream_wait_on_all(main_stream_, other_streams_...);
  }

  ~MultiSync()
  {
    std::apply(
        [&](auto&&... args) { all_wait_on_stream(main_stream, args...); },
        other_streams);
  }

  /**
   * Allow implicit conversion to the main compute stream.
   *
   * This enables a MultiSync to be passed in place of the main stream.
   */
  operator const ComputeStream&() const H2_NOEXCEPT { return main_stream; }

  /** Return the main stream associated with this MultiSync. */
  ComputeStream get_main_stream() const H2_NOEXCEPT { return main_stream; }

  /** Return the underlying raw stream associated with the main stream. */
  template <Device ThisDev>
  typename internal::RawComputeStream<ThisDev>::type
  get_stream() const H2_NOEXCEPT
  {
    return main_stream.get_stream<ThisDev>();
  }

private:
  ComputeStream main_stream;
  std::tuple<OtherStreams...> other_streams;
};

/** Create a new MultiSync. */
template <typename... OtherStreams>
MultiSync<OtherStreams...>
create_multi_sync(const ComputeStream& main,
                  const OtherStreams&... other_streams)
{
  return MultiSync(main, other_streams...);
}

}  // namespace h2

namespace std
{

// Inject hash specialization for sync objects.

template <>
struct hash<h2::SyncEvent>
{
  size_t operator()(const h2::SyncEvent& event) const H2_NOEXCEPT
  {
    using h2::Device;
    H2_DEVICE_DISPATCH(
      event.get_device(),
      return 0,
      return hash<void*>()((void*) event.get_event<Dev>()));
  }
};

template <>
struct hash<h2::ComputeStream>
{
  size_t operator()(const h2::ComputeStream& stream) const H2_NOEXCEPT
  {
    using h2::Device;
    H2_DEVICE_DISPATCH(
      stream.get_device(),
      return 0,
      return hash<void*>()((void*) stream.get_stream<Dev>()));
  }
};

}  // namespace std
