#include <h2/memory/gpu/cub_allocator_impl.hpp>

#include <h2/gpu/memory_utils.hpp>
#include <h2/gpu/runtime.hpp>

#include <h2/memory/allocator.hpp>

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Testing the CUBAllocator", "[gpu][memory]",
                   int, long, float, double)
{
    using T = TestType;

    if (!h2::gpu::runtime_is_initialized())
        h2::gpu::init_runtime();

    h2::gpu::CUBAllocator<T> alloc;
    SECTION("Streamless allocation")
    {
        size_t const size = 16UL;
        T* const ptr = h2::allocate<T>(alloc, size);

        CHECK(ptr);

        CHECK_NOTHROW(h2::gpu::mem_zero(ptr, size));

        std::vector<T> cpu_mem(size, static_cast<T>(-1));
        CHECK_NOTHROW(h2::gpu::mem_copy(cpu_mem.data(), ptr, size));
        CHECK(cpu_mem == std::vector<T>(size, static_cast<T>(0)));

        alloc.deallocate(ptr, size);
        h2::gpu::sync();
    }

    SECTION("Streamy allocation")
    {
        size_t const size = 16UL;
        auto stream = h2::gpu::make_stream();
        T* const ptr = h2::allocate<T>(alloc, size, stream);

        CHECK(ptr);

        CHECK_NOTHROW(h2::gpu::mem_zero(ptr, size, stream));

        std::vector<T> cpu_mem(size, static_cast<T>(-1));
        CHECK_NOTHROW(h2::gpu::mem_copy(cpu_mem.data(), ptr, size, stream));
        CHECK_NOTHROW(h2::gpu::sync(stream));

        CHECK(cpu_mem == std::vector<T>(size, static_cast<T>(0)));

        alloc.deallocate(ptr, size, stream);
        h2::gpu::sync(stream);
        h2::gpu::destroy(stream);
    }

}
