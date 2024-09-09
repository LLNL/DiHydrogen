#include "distconv/tensor/runtime_cuda.hpp"

#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"

#include <algorithm>

#include <cuda_runtime.h>

namespace distconv
{
namespace tensor
{
namespace internal
{

PinnedMemoryPool::PinnedMemoryPool()
{}
PinnedMemoryPool::~PinnedMemoryPool()
{
  deallocate_all_chunks();
}

void* PinnedMemoryPool::get(size_t size)
{
  for (auto it = m_chunks.begin(); it != m_chunks.end(); ++it)
  {
    chunk_t& c = *it;
    if (std::get<1>(c) >= size && !std::get<2>(c))
    {
      std::get<2>(c) = true;
      return std::get<0>(c);
    }
  }
  util::PrintStreamDebug() << "Allocating a new pinned memory of size " << size
                           << "\n";
  void* new_mem;
  DISTCONV_CHECK_CUDA(cudaMallocHost(&new_mem, size));
  chunk_t c = std::make_tuple(new_mem, size, true);
  m_chunks.emplace_back(c);
  return std::get<0>(c);
}

void PinnedMemoryPool::release(void* p)
{
  for (auto it = m_chunks.begin(); it != m_chunks.end(); ++it)
  {
    chunk_t& c = *it;
    if (std::get<0>(c) == p)
    {
      assert_always(std::get<2>(c) && "Error: Releasing unused pointer");
      std::get<2>(c) = false;
      return;
    }
  }
  assert_always(false && "Error: Releasing unknown pointer");
}

void PinnedMemoryPool::deallocate_all_chunks()
{
  std::for_each(m_chunks.begin(), m_chunks.end(), [](chunk_t c) {
    assert_always(!std::get<2>(c));
    cudaFreeHost(std::get<0>(c));
  });
  m_chunks.clear();
}

RuntimeCUDA* RuntimeCUDA::m_instance = nullptr;

RuntimeCUDA::RuntimeCUDA()
{}

RuntimeCUDA& RuntimeCUDA::get_instance()
{
  if (m_instance == nullptr)
  {
    m_instance = new RuntimeCUDA();
  }
  return *m_instance;
}

PinnedMemoryPool& RuntimeCUDA::get_pinned_memory_pool()
{
  return get_instance().m_pmp;
}

}  // namespace internal
}  // namespace tensor
}  // namespace distconv
