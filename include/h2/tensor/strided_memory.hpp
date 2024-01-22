////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Manages memory and an associated stride.
 */

#include <memory>
#include <utility>
#include <cstddef>

#include "h2/tensor/tensor_types.hpp"
#include "h2/tensor/raw_buffer.hpp"

namespace h2 {

/**
 * Return the strides needed for a tensor of the given shape to be
 * contiguous.
 */
constexpr inline StrideTuple get_contiguous_strides(ShapeTuple shape) {
  StrideTuple strides(TuplePad<StrideTuple>(shape.size(), 1));
  // Just need a prefix-product.
  for (typename ShapeTuple::size_type i = 1; i < shape.size(); ++i) {
    strides[i] = shape[i-1] * strides[i-1];
  }
  return strides;
}

/**
 * Return true if the given strides are contiguous.
 */
constexpr inline bool are_strides_contiguous(
  ShapeTuple shape,
  StrideTuple strides) {
  H2_ASSERT_DEBUG(shape.size() == strides.size(),
                  "Shape and strides must be the same size");
  // Contiguous strides should follow the prefix-product.
  typename StrideTuple::type prod = 1;
  typename ShapeTuple::size_type i = 1;
  for (; i < shape.size(); ++i)
  {
    if (prod != strides[i-1])
    {
      return false;
    }
    prod *= shape[i-1];
  }
  // Check the last entry and handle empty tuples.
  return (strides.size() == 0) || (prod == strides[i-1]);
}

/**
 * A managed chunk of memory with an associated stride.
 */
template <typename T, Device Dev>
class StridedMemory
{
private:
  static constexpr std::size_t INVALID_OFFSET = static_cast<std::size_t>(-1);

  using raw_buffer_t = RawBuffer<T, Dev>;
public:

  /** Allocate empty memory. */
  StridedMemory(bool lazy = false, const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : raw_buffer(nullptr),
      mem_offset(INVALID_OFFSET),
      unowned_mem_buffer(nullptr),
      mem_strides{},
      mem_shape{},
      sync_info(sync),
      is_mem_lazy(lazy)
  {}

  /** Allocate memory for shape, with unit strides. */
  StridedMemory(ShapeTuple shape, bool lazy = false,
                const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : StridedMemory(lazy, sync)
  {
    if (!shape.empty())
    {
      mem_strides = get_contiguous_strides(shape);
      mem_shape = shape;
      if (!lazy)
      {
        ensure(false);  // Allocate memory.
      }
      mem_offset = 0;
    }
  }

  /** View a subregion of an existing memory region. */
  StridedMemory(const StridedMemory<T, Dev>& base, CoordTuple coords)
    : raw_buffer(base.raw_buffer),
      mem_offset(base.get_index(get_range_start(coords))),
      unowned_mem_buffer(nullptr),
      mem_strides(
        TuplePad<StrideTuple>(base.mem_strides.size())), // Will be resized.
      mem_shape(get_range_shape(coords, base.shape())),
      sync_info(base.sync_info),
      is_mem_lazy(base.is_lazy())
  {
    H2_ASSERT_DEBUG(coords.size() <= base.mem_strides.size(),
                    "coords size not compatible with strides");
    typename StrideTuple::size_type j = 0;
    for (typename StrideTuple::size_type i = 0; i < base.mem_strides.size(); ++i) {
      // Dimensions that are now trivial are removed.
      // Unspecified dimensions on the right use their full range.
      if (i >= coords.size() || !is_coord_trivial(coords[i])) {
        mem_strides[j] = base.mem_strides[i];
        ++j;
      }
    }
    mem_strides.set_size(j);
  }

  /** Wrap an existing memory buffer. */
  StridedMemory(T* buffer,
                ShapeTuple shape,
                StrideTuple strides,
                const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : raw_buffer(nullptr),
      mem_offset(INVALID_OFFSET),
      unowned_mem_buffer(buffer),
      mem_strides(strides),
      mem_shape(shape),
      sync_info(sync),
      is_mem_lazy(false)
  {
    H2_ASSERT_DEBUG(buffer
                    || shape.empty()
                    || any_of(shape, [](ShapeTuple::type x) { return x == 0; }),
                    "Null buffer but non-zero shape provided to StridedMemory");
  }

  ~StridedMemory()
  {
    if (raw_buffer)
    {
      raw_buffer->register_release(sync_info);
    }
  }

  void ensure(bool attempt_recover = true)
  {
    if (unowned_mem_buffer || raw_buffer)
    {
      return;  // Data is already allocated.
    }
    if (attempt_recover)
    {
      // If the old raw buffer is still allocated, reuse it.
      raw_buffer = old_raw_buffer.lock();
    }
    if (!raw_buffer)
    {
      // Either not attempting to recover or no old raw buffer.
      // Do not allocate a RawBuffer for empty memory.
      if (!mem_shape.empty())
      {
        std::size_t size = product<std::size_t>(mem_shape);
        if (size) {
          raw_buffer = std::make_shared<raw_buffer_t>(size, sync_info);
        }
      }
    }
    old_raw_buffer.reset();  // Drop reference to old raw buffer.
  }

  void release()
  {
    if (unowned_mem_buffer)
    {
      unowned_mem_buffer = nullptr;
    }
    else if (raw_buffer)
    {
      old_raw_buffer = raw_buffer;
      raw_buffer.reset();
    }
  }

  T* data() H2_NOEXCEPT
  {
    return const_cast<T*>(std::as_const(*this).data());
  }

  const T* data() const H2_NOEXCEPT
  {
    return const_data();
  }

  const T* const_data() const H2_NOEXCEPT
  {
    if (unowned_mem_buffer)
    {
      return unowned_mem_buffer;
    }
    if (raw_buffer)
    {
      H2_ASSERT_DEBUG(mem_offset != INVALID_OFFSET,
                      "Invalid offset in StridedMemory: "
                      "raw_buffer is non-null but no offset was set");
      return raw_buffer->data() + mem_offset;
    }
    return nullptr;
  }

  StrideTuple strides() const H2_NOEXCEPT { return mem_strides; }

  typename StrideTuple::type stride(typename StrideTuple::size_type i)
  {
    return mem_strides[i];
  }

  ShapeTuple shape() const H2_NOEXCEPT { return mem_shape; }

  typename ShapeTuple::type shape(typename ShapeTuple::size_type i)
  {
    return mem_shape[i];
  }

  /** Get the index of coords in the buffer. */
  DataIndexType get_index(const SingleCoordTuple& coords) const H2_NOEXCEPT {
    return inner_product<DataIndexType>(coords, mem_strides);
  }

  /**
   * Return the coordinate corresponding to the given index in the
   * generalized column-major order.
   */
  SingleCoordTuple get_coord(DataIndexType idx) const H2_NOEXCEPT
  {
    SingleCoordTuple coord(TuplePad<SingleCoordTuple>(mem_shape.size()));
    for (typename ShapeTuple::size_type i = 0; i < mem_shape.size(); ++i)
    {
      coord[i] = (idx / mem_strides[i]) % mem_shape[i];
    }
    return coord;
  }

  /** Return a pointer to the memory at the given coordinates. */
  T* get(SingleCoordTuple coords) H2_NOEXCEPT {
    H2_ASSERT_DEBUG(data(), "No memory");
    return &(data()[get_index(coords)]);
  }

  const T* get(SingleCoordTuple coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(data(), "No memory");
    return &(data()[get_index(coords)]);
  }

  const T* const_get(SingleCoordTuple coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(const_data(), "No memory");
    return &(const_data()[get_index(coords)]);
  }

  SyncInfo<Dev> get_sync_info() const H2_NOEXCEPT
  {
    return sync_info;
  }

  void set_sync_info(const SyncInfo<Dev>& sync, bool set_raw = false)
  {
    sync_info = sync;
    if (raw_buffer && set_raw)
    {
      raw_buffer->set_sync_info(sync);
    }
  }

  bool is_lazy() const H2_NOEXCEPT
  {
    return is_mem_lazy;
  }

private:
  /**
   * Raw underlying memory buffer.
   *
   * This may be shared across StridedMemory objects that have different
   * strides or offsets into the raw buffer.
   *
   * If there is no memory, we are lazy and memory has not yet been
   * allocated, or we are wrapping externally managed memory, this is
   * null.
   *
   * As `RawBuffer` may change the underlying memory, when using one,
   * we work with offsets instead of direct pointers into it.
   */
  std::shared_ptr<raw_buffer_t> raw_buffer;
  std::size_t mem_offset;  /**< Offset to start of raw_buffer. */
  T* unowned_mem_buffer;  /**< Pointer to an externally managed  buffer. */
  /**
   * Reference to the prior `raw_buffer`, which may be used when
   * recovering existing memory from views using `ensure`.
   */
  std::weak_ptr<raw_buffer_t> old_raw_buffer;
  StrideTuple mem_strides;  /**< Strides associated with the memory. */
  ShapeTuple mem_shape;  /**< Shape describing the extent of the memory. */
  SyncInfo<Dev> sync_info;  /**< Synchronization info for operations. */
  bool is_mem_lazy;  /**< Whether allocation is lazy. */
};

/** Support printing StridedMemory. */
template <typename T, Device Dev>
inline std::ostream& operator<<(std::ostream& os,
                                const StridedMemory<T, Dev>& mem)
{
  // TODO: Print the type along with the device.
  os << "StridedMemory<" << Dev << ">("
     << (mem.is_lazy() ? "lazy" : "not lazy") << ", "
     << mem.data() << ", " << mem.strides()
     << ", " << mem.shape() << ")";
  return os;
}

/** Print the contents of StridedMemory. */
template <typename T, Device Dev>
inline std::ostream& strided_memory_contents(std::ostream& os,
                                             const StridedMemory<T, Dev>& mem)
{
  DataIndexType size =
      mem.shape().size() ? product<DataIndexType>(mem.shape()) : 0;
  for (DataIndexType i = 0; i < size; ++i)
  {
    os << *mem.get(mem.get_coord(i));
    if (i != size - 1)
    {
      os << ", ";
    }
  }
  return os;
}

}  // namespace h2
