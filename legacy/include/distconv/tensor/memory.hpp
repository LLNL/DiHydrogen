#pragma once

#include "distconv/tensor/stream.hpp"
#include "distconv/util/util.hpp"

#include <cstring>
#include <memory>
#include <sstream>
#include <type_traits>

namespace distconv
{
namespace tensor
{

template <bool b, typename T>
struct add_const;

template <typename T>
struct add_const<true, T>
{
  using type = const T;
};

template <typename T>
struct add_const<false, T>
{
  using type = T;
};

struct MemoryProperty
{
  size_t m_size;
  size_t m_ldim;
  size_t m_pitch;
  MemoryProperty() : m_size(0), m_ldim(0), m_pitch(0) {}
  MemoryProperty(size_t size, size_t ldim, size_t pitch)
    : m_size(size), m_ldim(ldim), m_pitch(pitch)
  {}
  void clear()
  {
    m_size = 0;
    m_ldim = 0;
    m_pitch = 0;
  }
  std::ostream& print(std::ostream& os) const
  {
    std::stringstream ss;
    ss << "("
       << "size: " << m_size << ", ldim: " << m_ldim << ", pitch: " << m_pitch
       << ")";
    os << ss.str();
    return os;
  }
};

inline std::ostream& operator<<(std::ostream& os, const MemoryProperty& mp)
{
  return mp.print(os);
}

template <typename Allocator>
class Memory
{
  // friend class Memory<Allocator, !is_const>;
public:
  // using element_type = typename add_const<is_const, void>::type;
  using allocator_type = Allocator;
#if 0
  using allocator_type = void (*)(void*&, size_t&, size_t, size_t);
  using deleter_type = void (*)(void *);
  using memsetter_type = void (*)(void*, size_t, int, size_t, size_t);
  using copyinout_type = void (*)(void *, const void *, size_t,
                                  size_t, size_t);
#endif

  Memory()
    : m_property(std::make_shared<MemoryProperty>()),
      m_managed_ptr(nullptr),
      m_alias_ptr(nullptr),
      m_alias_const_ptr(nullptr)
  {}

  ~Memory() = default;

  Memory(const Memory<Allocator>& m)
    : m_property(m.m_property),
      m_managed_ptr(m.m_managed_ptr),
      m_alias_ptr(m.m_alias_ptr),
      m_alias_const_ptr(m.m_alias_const_ptr)
  {}

  Memory(Memory&& m) : Memory() { swap(*this, m); }

  Memory<Allocator>& operator=(Memory<Allocator> m)
  {
    swap(*this, m);
    return *this;
  }

  friend void swap(Memory<Allocator>& x, Memory<Allocator>& y)
  {
    using std::swap;
    swap(x.m_property, y.m_property);
    swap(x.m_managed_ptr, y.m_managed_ptr);
    swap(x.m_alias_ptr, y.m_alias_ptr);
    swap(x.m_alias_const_ptr, y.m_alias_const_ptr);
  }

  void nullify()
  {
    m_managed_ptr.reset();
    m_property.reset();
    m_alias_ptr = nullptr;
    m_alias_const_ptr = nullptr;
  }

#if 0
  template <bool is_const>
  typename add_const<is_const, void>::type *get() const {
    using element_type = typename add_const<is_const, void>::type;
    if (m_const && !is_const) {
      return nullptr;
    }
    if (is_const && m_alias_const_ptr) {
      return m_alias_const_ptr;
    } else if (m_alias_ptr) {
      return static_cast<element_type*>(m_alias_ptr);
    } else {
      return static_cast<element_type*>(m_managed_ptr.get());
    }
  }
#endif

  const void* get() const
  {
    if (m_alias_const_ptr)
    {
      return m_alias_const_ptr;
    }
    else if (m_alias_ptr)
    {
      return m_alias_ptr;
    }
    else
    {
      return m_managed_ptr.get();
    }
  }

  void* get()
  {
    if (m_alias_ptr)
    {
      return m_alias_ptr;
    }
    else
    {
      return m_managed_ptr.get();
    }
  }

  bool is_null() const { return get() == nullptr; }

  bool is_non_null() const { return !is_null(); }

  size_t get_size() const { return m_property->m_size; }

  size_t get_ldim() const { return m_property->m_ldim; }

  size_t get_pitch() const { return m_property->m_pitch; }

  size_t get_real_size() const { return get_size() / get_ldim() * get_pitch(); }

  bool is_pitched() const { return get_pitch() != get_ldim(); }

  // Allocated object is always non const
  int allocate(size_t size, size_t ldim = 0)
  {
    if (size == 0)
    {
      std::cerr << "can't allocate empty object\n";
      return -1;
    }

    void* new_ptr = nullptr;
    size_t pitch;
    Allocator::allocate(new_ptr, pitch, size, ldim);
    if (new_ptr == nullptr)
    {
      std::cerr << "allocator returns null\n";
      return -1;
    }

    nullify();

    m_managed_ptr.reset(new_ptr, Allocator::deallocate);
    m_property = std::make_shared<MemoryProperty>(size, ldim, pitch);
    return 0;
  }

  // No use case
#if 0
  // Takes ownership
  int attach(element_type *ptr, size_t size, size_t ldim, size_t pitch=0) {
    nullify();
    m_managed_ptr.reset(ptr, m_deleter);
    m_property = std::make_shared<MemoryProperty>(size, ldim, pitch);
    return 0;
  }

  template <typename T>
  int attach(T *ptr, size_t size, size_t ldim, size_t pitch=0);
#endif

  int memset(
    int v,
    typename Stream<Allocator>::type stream = Stream<Allocator>::default_value)
  {
    void* dst = get();
    if (dst == nullptr)
      return -1;
    Allocator::memset(dst, get_pitch(), v, get_size(), get_ldim(), stream);
    return 0;
  }

  void copyout(void* p) const
  {
    Allocator::copyout(p, get(), get_size(), get_pitch(), get_ldim());
  }

  void copyin(const void* p)
  {
    void* dst = get();
    assert_always(dst != nullptr);
    Allocator::copyin(dst, p, get_size(), get_pitch(), get_ldim());
  }

  int alias(const Memory<Allocator>& m)
  {
    nullify();
    m_property = m.m_property;
    m_managed_ptr = m.m_managed_ptr;
    return 0;
  }

  int alias(void* ptr, size_t size, size_t ldim, size_t pitch)
  {
    nullify();
    m_alias_ptr = ptr;
    m_property = std::make_shared<MemoryProperty>(size, ldim, pitch);
    return 0;
  }

  int alias(const void* ptr, size_t size, size_t ldim, size_t pitch)
  {
    nullify();
    m_alias_const_ptr = ptr;
    m_property = std::make_shared<MemoryProperty>(size, ldim, pitch);
    return 0;
  }

  std::ostream& print(std::ostream& os) const
  {
    std::stringstream ss;
    ss << "(" << *m_property << ", managed ptr: " << m_managed_ptr.get()
       << ", alias ptr: " << m_alias_ptr << ", alias const ptr: " << m_alias_ptr
       << ")";
    os << ss.str();
    return os;
  }

protected:
  std::shared_ptr<MemoryProperty> m_property;
  // Keep allocated memory as non-const
  // std::shared_ptr<element_type> m_managed_ptr;
  std::shared_ptr<void> m_managed_ptr;
  void* m_alias_ptr;
  const void* m_alias_const_ptr;
};

template <typename Allocator>
inline std::ostream& operator<<(std::ostream& os, const Memory<Allocator>& m)
{
  return m.print(os);
}

struct BaseAllocator
{
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    size_t align_threshold = 1024 * 1024;
    if (size < align_threshold)
    {
      p = std::malloc(size);
    }
    else
    {
      p = util::aligned_malloc(size);
    }
    pitch = ldim;
  }
  static void deallocate(void* p) { std::free(p); }
  static void
  memset(void* p, size_t pitch, int v, size_t size, size_t, int stream = 0)
  {
    std::memset(p, v, size);
  }
  static void
  copyin(void* dst, const void* src, size_t real_size, size_t, size_t)
  {
    std::memcpy(dst, src, real_size);
  }
  static void
  copyout(void* dst, const void* src, size_t real_size, size_t, size_t)
  {
    std::memcpy(dst, src, real_size);
  }
};

template <int ALIGN_SIZE>
struct BasePitchedAllocator
{
  static constexpr int align_size = ALIGN_SIZE;
  static void allocate(void*& p, size_t& pitch, size_t size, size_t ldim)
  {
    pitch = (ldim / ALIGN_SIZE + ((ldim % ALIGN_SIZE) ? 1 : 0)) * ALIGN_SIZE;
    size_t pitched_size = size / ldim * pitch;
    p = std::malloc(pitched_size);
  }
  static void deallocate(void* p) { std::free(p); }
  static void
  memset(void* p, size_t pitch, int v, size_t size, size_t ldim, int stream = 0)
  {
    std::memset(p, v, pitch * size / ldim);
  }
  static void
  copyin(void* dst, const void* src, size_t size, size_t pitch, size_t ldim)
  {
    size_t nrows = size / ldim;
    size_t dst_offset = 0;
    size_t src_offset = 0;
    for (size_t i = 0; i < nrows; ++i)
    {
      std::memcpy((char*) dst + dst_offset, (char*) src + src_offset, ldim);
      dst_offset += pitch;
      src_offset += ldim;
    }
  }
  static void
  copyout(void* dst, const void* src, size_t size, size_t pitch, size_t ldim)
  {
    size_t nrows = size / ldim;
    size_t dst_offset = 0;
    size_t src_offset = 0;
    for (size_t i = 0; i < nrows; ++i)
    {
      std::memcpy((char*) dst + dst_offset, (char*) src + src_offset, ldim);
      src_offset += pitch;
      dst_offset += ldim;
    }
  }
};

template <>
struct Stream<BaseAllocator>
{
  using type = int;
  static constexpr type default_value = 0;
};

// template <>
template <int ALIGN_SIZE>
struct Stream<BasePitchedAllocator<ALIGN_SIZE>>
{
  using type = int;
  static constexpr type default_value = 0;
};

// Couldn't support Pitched memory
//  std::is_same<AllocDst, BasePitchedAllocator<PitchDst>>::value ||
//  std::is_same<AllocSrc, BasePitchedAllocator<PitchSrc>>::value
template <typename AllocDst,
          typename AllocSrc,
          typename StreamType = DefaultStream>
inline
  typename std::enable_if<std::is_same<AllocDst, BaseAllocator>::value
                            && std::is_same<AllocSrc, BaseAllocator>::value,
                          int>::type
  Copy(Memory<AllocDst>& dst,
       const Memory<AllocSrc>& src,
       size_t x_len,
       size_t y_len,
       size_t x_dst_offset,
       size_t y_dst_offset,
       size_t x_src_offset,
       size_t y_src_offset,
       StreamType stream = DefaultStream::value)
{
#if 0
  util::PrintStreamDebug()
      << "Copying memory of BaseAllocator: "
      << x_len << "x" << y_len << ", "
      << x_dst_offset << "x" << y_dst_offset << ", "
      << x_src_offset << "x" << y_src_offset << ", "
      << dst.get_pitch() << ", " << src.get_pitch() << "\n";
#endif
  // This check should not be necessary, but just for sanity check
  assert_always(dst.get_pitch() != 0);
  assert_always(src.get_pitch() != 0);
  for (size_t y = 0; y < y_len; ++y)
  {
    size_t src_offset = x_src_offset + src.get_pitch() * (y_src_offset + y);
    size_t dst_offset = x_dst_offset + dst.get_pitch() * (y_dst_offset + y);
    std::memcpy(
      (char*) dst.get() + dst_offset, (char*) src.get() + src_offset, x_len);
  }
  return 0;
}

}  // namespace tensor
}  // namespace distconv
