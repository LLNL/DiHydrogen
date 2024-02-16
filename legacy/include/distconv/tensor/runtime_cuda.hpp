#pragma once

#include <tuple>
#include <list>
#include <cstddef>

namespace distconv {
namespace tensor {
namespace internal {

// REFACTORING: Move this class to distconv::RuntimeCUDA
class PinnedMemoryPool {
 public:
  PinnedMemoryPool();
  ~PinnedMemoryPool();
  void *get(size_t size);
  void release(void *p);

 protected:
  using chunk_t = std::tuple<void*, size_t, bool>;
  std::list<chunk_t> m_chunks;

  void deallocate_all_chunks();
};

struct MemoryPoolChunk {
  using chunk_t = std::tuple<void*, size_t, bool>;
 public:
  MemoryPoolChunk():
      m_chunk(std::make_tuple(nullptr, 0, false)) {}
  MemoryPoolChunk(void *ptr, size_t s, bool t):
      m_chunk(std::make_tuple(ptr, s, t)) {}

  void *&pointer() {
    return std::get<0>(m_chunk);
  }

  void * const &pointer() const {
    return std::get<0>(m_chunk);
  }

  size_t &size() {
    return std::get<1>(m_chunk);
  }

  const size_t &size() const {
    return std::get<1>(m_chunk);
  }

  bool &taken() {
    return std::get<2>(m_chunk);
  }

  const bool &taken() const {
    return std::get<2>(m_chunk);
  }

  void set_taken(bool f) {
    taken() = f;
  }

  protected:
  chunk_t m_chunk;
};

class RuntimeCUDA {
 public:
  static PinnedMemoryPool &get_pinned_memory_pool();

 protected:
  static RuntimeCUDA *m_instance;
  PinnedMemoryPool m_pmp;

  RuntimeCUDA();
  static RuntimeCUDA &get_instance();
};

} // namespace internal
} // namespace tensor
} // namespace distconv
