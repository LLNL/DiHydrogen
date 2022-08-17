#pragma once

#include <list>
#include <tuple>

namespace distconv
{
namespace tensor
{
namespace internal
{

// REFACTORING: Move this class to distconv::RuntimeCUDA
class PinnedMemoryPool
{
public:
    PinnedMemoryPool();
    ~PinnedMemoryPool();
    void* get(size_t size);
    void release(void* p);

protected:
    using chunk_t = std::tuple<void*, size_t, bool>;
    std::list<chunk_t> m_chunks;

    void deallocate_all_chunks();
};

// struct MemoryPoolChunk
// {
//     using chunk_t = std::tuple<void*, size_t, bool>;

// public:
//     MemoryPoolChunk() : m_chunk(std::make_tuple(nullptr, 0, false)) {}
//     MemoryPoolChunk(void* ptr, size_t s, bool t)
//         : m_chunk(std::make_tuple(ptr, s, t))
//     {}

//     void*& pointer() noexcept { return std::get<0>(m_chunk); }

//     void* const& pointer() const noexcept { return std::get<0>(m_chunk); }

//     size_t& size() noexcept { return std::get<1>(m_chunk); }

//     size_t const& size() const noexcept { return std::get<1>(m_chunk); }

//     bool& taken() noexcept { return std::get<2>(m_chunk); }

//     bool const& taken() const noexcept { return std::get<2>(m_chunk); }

//     void set_taken(bool f) noexcept { taken() = f; }

// private:
//     chunk_t m_chunk;
// };

class RuntimeHIP
{
public:
    static PinnedMemoryPool& get_pinned_memory_pool();

private:
    RuntimeHIP();
    static RuntimeHIP& get_instance();

    PinnedMemoryPool m_pmp;
};

} // namespace internal
} // namespace tensor
} // namespace distconv
