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
class StridedMemory {
public:

  /** Allocate empty memory. */
  StridedMemory(const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : raw_buffer(nullptr),
      mem_buffer(nullptr),
      mem_strides{},
      mem_shape{},
      sync_info(sync)
  {}

  /** Allocate memory for shape, with unit strides. */
  StridedMemory(ShapeTuple shape, const SyncInfo<Dev>& sync = SyncInfo<Dev>{})
    : StridedMemory(sync)
  {
    if (!shape.empty())
    {
      mem_strides = get_contiguous_strides(shape);
      mem_shape = shape;
      raw_buffer = std::make_shared<RawBuffer<T, Dev>>(
        product<std::size_t>(shape), sync);
      mem_buffer = raw_buffer->data();
    }
  }

  /** View a subregion of an existing memory region. */
  StridedMemory(const StridedMemory<T, Dev>& base, CoordTuple coords)
    : raw_buffer(base.raw_buffer),
      mem_buffer(const_cast<T*>(base.get(get_range_start(coords)))),
      mem_strides(
        TuplePad<StrideTuple>(base.mem_strides.size())), // Will be resized.
      mem_shape(get_range_shape(coords, base.shape())),
      sync_info(base.sync_info)
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
      mem_buffer(buffer),
      mem_strides(strides),
      mem_shape(shape),
      sync_info(sync)
  {}

  ~StridedMemory()
  {
    if (raw_buffer)
    {
      raw_buffer->register_release(sync_info);
    }
  }

  T* data() H2_NOEXCEPT {
    return mem_buffer;
  }

  const T* data() const H2_NOEXCEPT {
    return mem_buffer;
  }

  const T* const_data() const H2_NOEXCEPT {
    return mem_buffer;
  }

  StrideTuple strides() const H2_NOEXCEPT { return mem_strides; }

  ShapeTuple shape() const H2_NOEXCEPT { return mem_shape; }

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
    H2_ASSERT_DEBUG(mem_buffer, "No memory");
    return &(mem_buffer[get_index(coords)]);
  }

  const T* get(SingleCoordTuple coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(mem_buffer, "No memory");
    return &(mem_buffer[get_index(coords)]);
  }

  const T* const_get(SingleCoordTuple coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(mem_buffer, "No memory");
    return &(mem_buffer[get_index(coords)]);
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

private:
  /**
   * Raw underlying memory buffer.
   *
   * This may be shared across StridedMemory objects that have different
   * strides or offsets into the raw buffer.
   */
  std::shared_ptr<RawBuffer<T, Dev>> raw_buffer;
  T* mem_buffer;  /**< Pointer to usable buffer. */
  StrideTuple mem_strides;  /**< Strides associated with the memory. */
  ShapeTuple mem_shape;  /**< Shape describing the extent of the memory. */
  SyncInfo<Dev> sync_info;  /**< Synchronization info for operations. */
};

/** Support printing StridedMemory. */
template <typename T, Device Dev>
inline std::ostream& operator<<(std::ostream& os,
                                const StridedMemory<T, Dev>& mem)
{
  // TODO: Print the type along with the device.
  os << "StridedMemory<" << Dev << ">(" << mem.data() << ", " << mem.strides()
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
