#include "p2p/util_cuda.hpp"

#include "p2p/logging.hpp"
#include "p2p/util.hpp"

namespace p2p
{
namespace util
{

PinnedMemoryPool::PinnedMemoryPool()
  : m_bin_growth(8), m_min_bin(3), m_max_bin(12)
{
  setup_bins();
}

PinnedMemoryPool::~PinnedMemoryPool()
{
  deallocate_all_chunks();
}

void PinnedMemoryPool::setup_bins()
{
  m_bins.clear();
  for (int i = m_min_bin; i <= m_max_bin; ++i)
  {
    m_bins.push_back(std::list<void*>());
  }
}

namespace
{
static size_t pow(int x, int y)
{
  size_t v = 1;
  for (int i = 0; i < y; ++i)
  {
    v *= x;
  }
  return v;
}
} // namespace

int PinnedMemoryPool::find_bin(size_t size)
{
  // Finds the right bin
  size_t bin_size = pow(m_bin_growth, m_min_bin);
  int bin_idx = -1;
  for (int i = m_min_bin; i <= m_max_bin; ++i)
  {
    if (size <= bin_size)
    {
      bin_idx = i - m_min_bin;
      break;
    }
    bin_size *= m_bin_growth;
  }
  P2P_ASSERT_ALWAYS(bin_idx != -1 && "Requested size too large");
  return bin_idx;
}

void* PinnedMemoryPool::get_from_bin(int bin_idx)
{
  logging::MPIPrintStreamDebug() << "Bin idx: " << bin_idx << "\n";
  auto& ch_list = m_bins[bin_idx];
  std::unique_lock<std::mutex> lock(m_mutex);
  if (ch_list.size() > 0)
  {
    auto ch = ch_list.begin();
    void* m = *ch;
    ch_list.erase(ch);
    lock.unlock();
    return m;
  }
  void* new_mem = nullptr;
  size_t bin_size = pow(m_bin_growth, m_min_bin + bin_idx);
  logging::MPIPrintStreamDebug()
    << "Allocating new memory of size " << bin_size << "\n";
  P2P_CHECK_CUDA_ALWAYS(cudaMallocHost(&new_mem, bin_size));
  // new_mem = std::malloc(bin_size);
  m_mem_map.insert(std::make_pair(new_mem, bin_size));
  lock.unlock();
  return new_mem;
}

void* PinnedMemoryPool::get(size_t size)
{
  int bin_idx = find_bin(size);
  return get_from_bin(bin_idx);
}

void PinnedMemoryPool::release(void* p)
{
  size_t size = m_mem_map.find(p)->second;
  int bin_idx = find_bin(size);
  std::unique_lock<std::mutex> lock(m_mutex);
  m_bins[bin_idx].push_back(p);
  lock.unlock();
  return;
}

void PinnedMemoryPool::deallocate_all_chunks()
{
  std::unique_lock<std::mutex> lock(m_mutex);
  for (auto& x : m_mem_map)
  {
    cudaFreeHost(x.first);
    // std::free(x.first);
  }
  m_mem_map.clear();
  setup_bins();
}

EventPool::EventPool(int num_events, int expansion) : m_expansion(expansion)
{
  EventPool::expand_list(m_events, num_events);
}

EventPool::~EventPool()
{
  for (auto x : m_events)
  {
    P2P_CHECK_CUDA_ALWAYS(cudaEventDestroy(x));
  }
  m_events.clear();
}

void EventPool::expand_list(std::list<cudaEvent_t>& list, int num_events)
{
  for (int i = 0; i < num_events; ++i)
  {
    cudaEvent_t e;
    P2P_CHECK_CUDA_ALWAYS(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    list.push_back(e);
  }
}

void EventPool::expand()
{
  EventPool::expand_list(m_events, m_expansion);
}

cudaEvent_t EventPool::get()
{
  std::unique_lock<std::mutex> lock(m_mutex);
  if (m_events.empty())
    expand();
  auto e = m_events.front();
  m_events.pop_front();
  lock.unlock();
  return e;
}

void EventPool::release(cudaEvent_t e)
{
  std::unique_lock<std::mutex> lock(m_mutex);
  m_events.push_back(e);
  lock.unlock();
}

size_t get_available_memory()
{
  size_t available;
  size_t total;
  P2P_CHECK_CUDA(cudaMemGetInfo(&available, &total));
  return available;
}

size_t get_total_memory()
{
  size_t available;
  size_t total;
  P2P_CHECK_CUDA(cudaMemGetInfo(&available, &total));
  return total;
}

} // namespace util
} // namespace p2p
