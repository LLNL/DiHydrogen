////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "distconv/tensor/runtime_rocm.hpp"

#include "distconv/util/util.hpp"
#include "distconv/util/util_rocm.hpp"

#include <algorithm>

#include <hip/hip_runtime.h>

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
  DISTCONV_CHECK_HIP(hipHostMalloc(&new_mem, size));
  chunk_t& c = m_chunks.emplace_back(new_mem, size, true);
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
    static_cast<void>(hipHostFree(std::get<0>(c)));
  });
  m_chunks.clear();
}

RuntimeHIP::RuntimeHIP()
{}

RuntimeHIP& RuntimeHIP::get_instance()
{
  // Note: Cannot use make_unique because it has no access to the
  // private ctor of the runtime class.
  static auto instance = std::unique_ptr<RuntimeHIP>(new RuntimeHIP);
  return *instance;
}

PinnedMemoryPool& RuntimeHIP::get_pinned_memory_pool()
{
  return get_instance().m_pmp;
}

} // namespace internal
} // namespace tensor
} // namespace distconv
