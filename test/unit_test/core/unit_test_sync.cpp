////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <unordered_map>

#include <El.hpp>

#include "h2/core/sync.hpp"

using namespace h2;

// GPU syncs, and GPU+CPU sync interactions tested in the GPU sync.

TEST_CASE("CPU SyncEvent works", "[sync]")
{
  SyncEvent<Device::CPU> event;

  REQUIRE(SyncEvent<Device::CPU>::device == Device::CPU);
  REQUIRE(event.get_device() == Device::CPU);
  REQUIRE_NOTHROW(event.wait_for_this());
}

TEST_CASE("CPU ComputeStream works", "[sync]")
{
  ComputeStream<Device::CPU> stream;
  SyncEvent<Device::CPU> event;

  REQUIRE(ComputeStream<Device::CPU>::device == Device::CPU);
  REQUIRE(stream.get_device() == Device::CPU);

  REQUIRE_NOTHROW(stream.add_sync_point(event));
  REQUIRE_NOTHROW(stream.wait_for(event));

  ComputeStream<Device::CPU> stream2;
  REQUIRE_NOTHROW(stream.wait_for(stream2));

  REQUIRE_NOTHROW(stream.wait_for_this());
}

TEST_CASE("CPU sync creation routines work", "[sync]")
{
  ComputeStream<Device::CPU> stream;
  REQUIRE_NOTHROW([&]() { stream = create_new_compute_stream<Device::CPU>(); }());
  REQUIRE_NOTHROW(destroy_compute_stream(stream));

  SyncEvent<Device::CPU> event;
  REQUIRE_NOTHROW([&]() { event = create_new_sync_event<Device::CPU>(); }());
  REQUIRE_NOTHROW(destroy_sync_event(event));
}

TEST_CASE("CPU sync helpers work", "[sync]")
{
  ComputeStream<Device::CPU> stream1, stream2, stream3;

  REQUIRE_NOTHROW(all_wait_on_stream(stream1, stream2, stream3));
  REQUIRE_NOTHROW(stream_wait_on_all(stream1, stream2, stream3));
}

TEST_CASE("CPU stream equality works", "[sync]")
{
  ComputeStream<Device::CPU> stream1 = create_new_compute_stream<Device::CPU>();
  ComputeStream<Device::CPU> stream2 = create_new_compute_stream<Device::CPU>();

  REQUIRE(stream1 == stream1);
  REQUIRE(stream1 == stream2);
  REQUIRE_FALSE(stream1 != stream2);

  std::unordered_map<ComputeStream<Device::CPU>, int> map;
  map[stream1] = 1;
  REQUIRE(map.count(stream1) > 0);
  REQUIRE(map.count(stream2) > 0);
}

TEST_CASE("CPU event equality works", "[sync]")
{
  SyncEvent<Device::CPU> event1 = create_new_sync_event<Device::CPU>();
  SyncEvent<Device::CPU> event2 = create_new_sync_event<Device::CPU>();

  REQUIRE(event1 == event2);
  REQUIRE(event1 == event2);
  REQUIRE_FALSE(event1 != event2);

  std::unordered_map<SyncEvent<Device::CPU>, int> map;
  map[event1] = 1;
  REQUIRE(map.count(event1) > 0);
  REQUIRE(map.count(event2) > 0);
}

TEST_CASE("CPU sync El::SyncInfo conversion works", "[sync]")
{
  // Conversion from El:
  El::SyncInfo<El::Device::CPU> sync_info =
      El::CreateNewSyncInfo<El::Device::CPU>();
  ComputeStream<Device::CPU> stream;
  REQUIRE_NOTHROW([&]() { stream = ComputeStream<Device::CPU>(sync_info); }());
  El::DestroySyncInfo(sync_info);

  // Conversion to El:
  REQUIRE_NOTHROW([&]() {
    sync_info = static_cast<El::SyncInfo<El::Device::CPU>>(stream);
  }());
}
