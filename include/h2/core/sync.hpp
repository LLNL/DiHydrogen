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

#include <El.hpp>

#include <tuple>

#include "h2/tensor/tensor_types.hpp"
#include "h2/utils/Error.hpp"


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

// Forward-declaration.
class SyncEventBase;

// Base classes without devices:

/**
 * Base class for device-specific compute streams.
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
class ComputeStreamBase
{
public:

  /** Return the device type of the stream. */
  virtual Device get_device() const H2_NOEXCEPT = 0;

  /** Record the current state of the stream in the given event. */
  virtual void add_sync_point(const SyncEventBase& event) const = 0;

  /** Have this stream wait for the provided event to complete. */
  virtual void wait_for(const SyncEventBase& event) const = 0;
  /**
   * Have this stream wait for all work currently on the given stream
   * to complete.
   *
   * This is equivalent to `wait_for(*stream.add_sync_point())`.
   */
  virtual void wait_for(const ComputeStreamBase& stream) const = 0;
  /**
   * Wait for all work currently on this stream to complete.
   *
   * This does nothing if this is a CPU stream. If this is a GPU
   * stream, the caller will wait for it.
   */
  virtual void wait_for_this() const = 0;
};

inline std::ostream& operator<<(std::ostream& os,
                                const ComputeStreamBase& stream)
{
  os << stream.get_device() << " compute stream";
  return os;
}

template <Device Dev>
class ComputeStream {};

/**
 * Base class for device-specific synchronization objects.
 *
 * These are used to synchronize between ComputeStreams.
 *
 * On CPUs, these are emtpy. On GPUs, these correspond to events.
 */
class SyncEventBase
{
public:

  /** Return the device type of the event. */
  virtual Device get_device() const H2_NOEXCEPT = 0;

  /**
   * Wait for all work currently recorded by the event to complete.
   *
   * This does nothing if this is a CPU event. If this is a GPU event,
   * the caller will wait for it.
   */
  virtual void wait_for_this() const = 0;
};

inline std::ostream& operator<<(std::ostream& os, const SyncEventBase& event)
{
  os << event.get_device() << " sync event";
  return os;
}

template <Device Dev>
class SyncEvent {};

// General utilities for interacting with compute streams and events:

/**
 * Have the main stream wait for all work currently on the other
 * streams to complete.
 */
template <Device Dev, Device... Devs>
inline void all_wait_on_stream(const ComputeStream<Dev>& main,
                               const ComputeStream<Devs>&... others)
{
  (main.wait_for(others), ...);
}

/**
 * Have the other streams wait for all work currently on the main
 * stream to complete.
 */
template <Device Dev, Device... Devs>
inline void stream_wait_on_all(const ComputeStream<Dev>& main,
                               const ComputeStream<Devs>&... others)
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
template <Device Dev, Device... Devs>
class MultiSync
{
public:
  MultiSync(const ComputeStream<Dev>& main_stream_,
            const ComputeStream<Devs>&... other_streams_)
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
  operator const ComputeStream<Dev>&() const H2_NOEXCEPT
  {
    return main_stream;
  }

private:
  ComputeStream<Dev> main_stream;
  std::tuple<ComputeStream<Devs>...> other_streams;
};

/** Create a fresh compute stream. */
template <Device Dev>
inline ComputeStream<Dev> create_new_compute_stream();

/** Destroy an existing compute stream. */
template <Device Dev>
inline void destroy_compute_stream(ComputeStream<Dev>& stream);

/** Create a fresh synchronization event. */
template <Device Dev>
inline SyncEvent<Dev> create_new_sync_event();

/** Destroy an existing synchronization event. */
template <Device Dev>
inline void destroy_sync_event(SyncEvent<Dev>& event);

/** Create a new MultiSync. */
template <Device Dev, Device... Devs>
MultiSync<Dev, Devs...>
create_multi_sync(const ComputeStream<Dev>& main,
                  const ComputeStream<Devs>&... other_streams)
{
  return MultiSync(main, other_streams...);
}

/**
 * RAII manager for events.
 */
template <Device Dev>
class SyncEventRAII
{
public:
  SyncEventRAII() : event(create_new_sync_event<Dev>()) {}

  ~SyncEventRAII() { destroy_sync_event(event); }

  // Prevent copying.
  SyncEventRAII(const SyncEventRAII&) = delete;
  SyncEventRAII<Dev>& operator=(const SyncEventRAII<Dev>&) = delete;

  SyncEventRAII(SyncEventRAII<Dev>&& other) { event = std::move(other.event); }
  SyncEventRAII<Dev>& operator=(SyncEventRAII<Dev>&& other)
  {
    event = std::move(other.event);
  }

  /** Allow conversion to a regular SyncEvent. */
  operator SyncEvent<Dev>() const H2_NOEXCEPT
  {
    return event;
  }

  SyncEvent<Dev> event;
};

// CPU compute streams and events:

template <>
class SyncEvent<Device::CPU> final : public SyncEventBase
{
public:
  static constexpr Device device = Device::CPU;

  Device get_device() const override H2_NOEXCEPT { return Device::CPU; }

  void wait_for_this() const override {}
};

/** Equality for CPU events. */
inline bool operator==(const SyncEvent<Device::CPU>&,
                       const SyncEvent<Device::CPU>) H2_NOEXCEPT
{
  return true;
}

/** Inequality for CPU events. */
inline bool operator!=(const SyncEvent<Device::CPU>&,
                       const SyncEvent<Device::CPU>&) H2_NOEXCEPT
{
  return false;
}

template <>
class ComputeStream<Device::CPU> final : public ComputeStreamBase
{
public:
  static constexpr Device device = Device::CPU;

  ComputeStream() H2_NOEXCEPT {}

  /** Support conversion from El::SyncInfo. */
  explicit ComputeStream(const El::SyncInfo<Device::CPU>&) H2_NOEXCEPT {}

  /** Support conversion to El::SyncInfo. */
  explicit operator El::SyncInfo<Device::CPU>() const H2_NOEXCEPT
  {
    return El::SyncInfo<Device::CPU>();
  }

  Device get_device() const override H2_NOEXCEPT { return Device::CPU; }

  void add_sync_point(const SyncEventBase& event) const override
  {
    H2_ASSERT_DEBUG(event.get_device() == Device::CPU,
                    "SyncEvent for CPU stream must be a CPU type");
    // Nothing to do, but we do sanity-check this.
  }

  void wait_for(const SyncEventBase& event) const override
  {
    // If this is a CPU event, the wait will do nothing.
    // If it is a GPU event, this will do the wait.
    event.wait_for_this();
  }

  void wait_for(const SyncEvent<Device::CPU>& event) const H2_NOEXCEPT {}

  void wait_for(const ComputeStreamBase& stream) const override
  {
    // If this is a CPU stream, the wait will do nothing.
    // If this is a GPU stream, this will do the wait.
    stream.wait_for_this();
  }

  void wait_for(const ComputeStream<Device::CPU>& stream) const H2_NOEXCEPT {}

  void wait_for_this() const override {}
};

/** Equality for CPU streams. */
inline bool operator==(const ComputeStream<Device::CPU>&,
                       const ComputeStream<Device::CPU>&) H2_NOEXCEPT
{
  return true;
}

/** Inequality for CPU streams. */
inline bool operator!=(const ComputeStream<Device::CPU>&,
                       const ComputeStream<Device::CPU>&) H2_NOEXCEPT
{
  return false;
}

template <>
inline ComputeStream<Device::CPU> create_new_compute_stream<Device::CPU>()
{
  return ComputeStream<Device::CPU>{};
}

template <>
inline void destroy_compute_stream<Device::CPU>(ComputeStream<Device::CPU>&) {}

template <>
inline SyncEvent<Device::CPU> create_new_sync_event<Device::CPU>()
{
  return SyncEvent<Device::CPU>{};
}

template <>
inline void destroy_sync_event<Device::CPU>(SyncEvent<Device::CPU>&) {}

}  // namespace h2

namespace std
{

// Inject hash specialization for CPU sync objects.

template <>
struct hash<h2::SyncEvent<h2::Device::CPU>>
{
  size_t operator()(const h2::SyncEvent<h2::Device::CPU>&) const noexcept
  {
    return 0;  // CPU syncs are all the same.
  }
};

template <>
struct hash<h2::ComputeStream<h2::Device::CPU>>
{
  size_t operator()(const h2::ComputeStream<h2::Device::CPU>&) const noexcept
  {
    return 0;  // CPU streams are all the same.
  }
};

}  // namespace std
