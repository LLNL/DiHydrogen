////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/gpu/sync.hpp"

using namespace h2;

// Currently, these tests do not really hit the knarly sync issues that
// could come up, at least partly because the GPU-side infra in H2 is
// currently pretty basic (and reliable tests of synchronization are
// hard to do). Maybe in the future we can improve this, especially if
// we discover bugs.

TEST_CASE("GPU SyncEvent works", "[sync]")
{
  SyncEvent<Device::GPU> event;

  REQUIRE(SyncEvent<Device::GPU>::device == Device::GPU);
  REQUIRE(event.get_device() == Device::GPU);
  REQUIRE(event.get_event() != nullptr);
  REQUIRE_NOTHROW(event.wait_for_this());
}

TEST_CASE("GPU ComputeStream works", "[sync]")
{
  ComputeStream<Device::GPU> stream;
  SyncEvent<Device::GPU> event;

  REQUIRE(ComputeStream<Device::GPU>::device == Device::GPU);
  REQUIRE(stream.get_device() == Device::GPU);

  REQUIRE(stream.get_stream() == El::cuda::GetDefaultStream());

  REQUIRE_NOTHROW(stream.add_sync_point(event));
  REQUIRE_NOTHROW(stream.wait_for(event));

  ComputeStream<Device::GPU> stream2;  // Note: Same underlying stream.
  REQUIRE_NOTHROW(stream.wait_for(stream2));

  REQUIRE_NOTHROW(stream.wait_for_this());
}

TEST_CASE("GPU sync creation routines work", "[sync]")
{
  ComputeStream<Device::GPU> stream;
  REQUIRE_NOTHROW([&]() { stream = create_new_compute_stream<Device::GPU>(); }());
  REQUIRE_NOTHROW(destroy_compute_stream(stream));

  SyncEvent<Device::GPU> event;
  REQUIRE_NOTHROW([&]() { event = create_new_sync_event<Device::GPU>(); }());
  REQUIRE_NOTHROW(destroy_sync_event(event));
}

TEST_CASE("GPU sync helpers work", "[sync]")
{
  ComputeStream<Device::GPU> stream1, stream2, stream3;
  stream2 = create_new_compute_stream<Device::GPU>();
  stream3 = create_new_compute_stream<Device::GPU>();
  ComputeStream<Device::CPU> stream4;

  REQUIRE_NOTHROW(all_wait_on_stream(stream1, stream2, stream3, stream4));
  REQUIRE_NOTHROW(stream_wait_on_all(stream1, stream2, stream3, stream4));
}

TEST_CASE("GPU and CPU syncs interoperate", "[sync]")
{
  ComputeStream<Device::GPU> gpu_stream;
  ComputeStream<Device::CPU> cpu_stream;
  SyncEvent<Device::GPU> gpu_event;
  SyncEvent<Device::CPU> cpu_event;

#ifdef H2_DEBUG
  REQUIRE_THROWS(gpu_stream.add_sync_point(cpu_event));
  REQUIRE_THROWS(cpu_stream.add_sync_point(gpu_event));
#endif

  REQUIRE_NOTHROW(gpu_stream.wait_for(cpu_event));
  REQUIRE_NOTHROW(cpu_stream.wait_for(gpu_event));

  REQUIRE_NOTHROW(gpu_stream.wait_for(cpu_stream));
  REQUIRE_NOTHROW(cpu_stream.wait_for(gpu_stream));

  REQUIRE_NOTHROW([&]() {
    auto multi_sync = create_multi_sync(gpu_stream, cpu_stream);
  }());
}

TEST_CASE("GPU sync El::SyncInfo conversion works", "[sync]")
{
  // Conversion from El:
  El::SyncInfo<El::Device::GPU> sync_info =
      El::CreateNewSyncInfo<Device::GPU>();
  ComputeStream<Device::GPU> stream;
  REQUIRE_NOTHROW([&]() { stream = ComputeStream<Device::GPU>(sync_info); }());
  REQUIRE(stream.get_stream() == sync_info.Stream());
  El::DestroySyncInfo(sync_info);

  // Conversion to El:
  REQUIRE_NOTHROW([&]() {
    sync_info = static_cast<El::SyncInfo<El::Device::GPU>>(stream);
  }());
  REQUIRE(sync_info.Stream() == stream.get_stream());
}
