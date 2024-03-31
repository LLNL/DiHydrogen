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

// CPU compute streams and events:

template <>
class SyncEvent<Device::CPU> final : public SyncEventBase
{
public:
  static constexpr Device device = Device::CPU;

  Device get_device() const override H2_NOEXCEPT { return Device::CPU; }

  void wait_for_this() const override {}
};

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
