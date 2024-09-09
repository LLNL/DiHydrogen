////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
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

class RuntimeHIP
{
public:
  static PinnedMemoryPool& get_pinned_memory_pool();

private:
  RuntimeHIP();
  static RuntimeHIP& get_instance();

  PinnedMemoryPool m_pmp;
};

}  // namespace internal
}  // namespace tensor
}  // namespace distconv
