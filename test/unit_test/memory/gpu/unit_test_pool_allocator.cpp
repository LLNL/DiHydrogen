#include <h2/memory/gpu/pool_allocator_impl.hpp>

#include <h2/gpu/memory_utils.hpp>
#include <h2/gpu/runtime.hpp>

#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("Testing the PoolAllocator", "[gpu][memory]",
                   int, long, float, double)
{
    using T = TestType;

    if (!h2::gpu::runtime_is_initialized())
        h2::gpu::init_runtime();

    h2::gpu::PoolAllocator<T> alloc;
    SECTION("Streamless allocation")
    {
        size_t const size = 16UL;
        T* const ptr = alloc.allocate(size);

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
        T* const ptr = alloc.allocate(size, stream);

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

TEMPLATE_TEST_CASE("Testing the PoolAllocator with external pool",
                   "[gpu][memory]",
                   int, long, float, double)
{
    using T = TestType;

    if (!h2::gpu::runtime_is_initialized())
        h2::gpu::init_runtime();

#if H2_HAS_CUDA
    cudaMemPool_t pool;
    cudaMemPoolProps props {
        .allocType = cudaMemAllocationTypePinned,
        .handleTypes = cudaMemHandleTypeNone,
        .location = {cudaMemLocationTypeDevice, h2::gpu::current_gpu()},
        .win32SecurityAttributes = nullptr,
        .reserved = {},
    };
    H2_CHECK_CUDA(cudaMemPoolCreate(&pool, &props));
#elif H2_HAS_ROCM
    hipMemPool_t pool;
    hipMemPoolProps props {
        .allocType = hipMemAllocationTypePinned,
        .handleTypes = hipMemHandleTypeNone,
        .location = {hipMemLocationTypeDevice, h2::gpu::current_gpu()},
        .win32SecurityAttributes = nullptr,
        .reserved = {},
    };
    H2_CHECK_HIP(hipMemPoolCreate(&pool, &props));
#endif

    h2::gpu::PoolAllocator<T> alloc(pool);
    SECTION("Streamless allocation")
    {
        size_t const size = 16UL;
        T* const ptr = alloc.allocate(size);

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
        T* const ptr = alloc.allocate(size, stream);

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

#if H2_HAS_CUDA
    H2_CHECK_CUDA(cudaMemPoolDestroy(pool));
#elif H2_HAS_ROCM
    H2_CHECK_HIP(hipMemPoolDestroy(pool));
#endif
}
