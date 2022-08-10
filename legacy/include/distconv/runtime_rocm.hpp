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
    void release(void* p);
    size_t get_max_allocatable_size(size_t limit);

private:
    hipcub::CachingDeviceAllocator m_allocator;
};

class RuntimeHIP
{
public:
    static HIPDeviceMemoryPool& get_device_memory_pool();
    static hipEvent_t& get_event(int idx = 0);

private:
    RuntimeHIP();
    static RuntimeHIP& get_instance();

    // PinnedMemoryPool m_pmp;
    HIPDeviceMemoryPool m_dmp;
    std::array<hipEvent_t, 2> m_events;
};

} // namespace internal
} // namespace distconv
