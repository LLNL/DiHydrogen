////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <array>

#include <hip/hip_runtime.h>
#include <hipcub/util_allocator.hpp>

namespace distconv
{
namespace internal
{

class HIPDeviceMemoryPool
{
public:
    HIPDeviceMemoryPool();
    ~HIPDeviceMemoryPool();
    void* get(size_t size, hipStream_t st);
    void release(void* p, hipStream_t st);
    size_t get_max_allocatable_size(size_t limit);
};

class RuntimeHIP
{
public:
    static HIPDeviceMemoryPool& get_device_memory_pool();
    static hipEvent_t& get_event(int idx = 0);
};

} // namespace internal
} // namespace distconv
