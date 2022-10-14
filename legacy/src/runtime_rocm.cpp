#include "distconv/runtime_rocm.hpp"

#include "distconv/util/util.hpp"
#include "distconv/util/util_rocm.hpp"

#include <algorithm>
#include <cmath>

#include <hip/hip_runtime.h>

namespace distconv
{
namespace internal
{

HIPDeviceMemoryPool::HIPDeviceMemoryPool()
    : m_allocator(
        /*bin_growth=*/4,
        /*min_bin=*/8,
        /*max_bin=*/hipcub::CachingDeviceAllocator::INVALID_BIN,
        /*max_cached_bytes=*/hipcub::CachingDeviceAllocator::INVALID_SIZE,
        /*skip_cleanup=*/false,
#ifdef DISTCONV_DEBUG
        /*debug=*/true
#else
        /*debug=*/false
#endif
    )
{}

HIPDeviceMemoryPool::~HIPDeviceMemoryPool()
{}

void* HIPDeviceMemoryPool::get(size_t size, hipStream_t st)
{
    void* p = nullptr;
    hipError_t const err = m_allocator.DeviceAllocate(&p, size, st);
    if (err != hipSuccess)
    {
        size_t available;
        size_t total;
        DISTCONV_CHECK_HIP(hipMemGetInfo(&available, &total));
        available /= (1024 * 1024);
        total /= (1024 * 1024);
        util::PrintStreamError()
            << "Allocation of " << size << " bytes (" << size / 1024.0 / 1024.0
            << " MB) failed. " << available << " MB available out of " << total
            << " MB.";
        std::abort();
    }
    assert_always(p);
    return p;
}

void HIPDeviceMemoryPool::release(void* p)
{
    DISTCONV_CHECK_HIP(m_allocator.DeviceFree(p));
}

size_t HIPDeviceMemoryPool::get_max_allocatable_size(size_t const limit)
{
    size_t const bin_growth = m_allocator.bin_growth;
    size_t const x = std::log(limit) / std::log(bin_growth);
    size_t const max_allowed_size = std::pow(bin_growth, x);
    return max_allowed_size;
}

namespace
{

struct RuntimeHIP_impl
{
    RuntimeHIP_impl();

    // PinnedMemoryPool m_pmp;
    HIPDeviceMemoryPool m_dmp;
    std::array<hipEvent_t, 2> m_events;
};

RuntimeHIP_impl::RuntimeHIP_impl()
{
    for (int i = 0; i < m_events.size(); ++i)
    {
        DISTCONV_CHECK_HIP(
            hipEventCreateWithFlags(&m_events[i], hipEventDisableTiming));
    }
}

RuntimeHIP_impl& get_runtime()
{
    static auto runtime = RuntimeHIP_impl{};
    return runtime;
}

} // namespace

HIPDeviceMemoryPool& RuntimeHIP::get_device_memory_pool()
{
    return get_runtime().m_dmp;
}

hipEvent_t& RuntimeHIP::get_event(int const idx)
{
    auto& rt = get_runtime();
    assert_always(idx < rt.m_events.size());
    return rt.m_events[idx];
}

} // namespace internal
} // namespace distconv
