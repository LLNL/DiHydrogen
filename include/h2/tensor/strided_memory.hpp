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
    strides[i] = shape[i] * strides[i-1];
  }
  return strides;
}

/**
 * Return true if the given strides are contiguous.
 */
constexpr inline bool are_strides_contiguous(StrideTuple strides) {
  // Ensure the strides follow the prefix-product.
  typename StrideTuple::type prod = 1;
  for (typename StrideTuple::size_type i = 0; i < strides.size() - 1; ++i) {
    if (strides[i] != prod) {
      return false;
    }
    prod *= strides[i];
  }
  return true;
}

/**
 * A managed chunk of memory with an associated stride.
 */
template <typename T, Device Dev>
class StridedMemory {
public:
  /** Allocate empty memory. */
  StridedMemory() :
    raw_buffer(nullptr),
    mem_buffer(nullptr),
    mem_strides{}
  {}

  /** Allocate memory for shape, with unit strides. */
  StridedMemory(ShapeTuple shape) : StridedMemory() {
    mem_strides = get_contiguous_strides(shape);
    raw_buffer = std::make_shared<RawBuffer<T, Dev>>(product<std::size_t>(shape));
    mem_buffer = raw_buffer->data();
  }

  /** View a subregion of an existing memory region. */
  StridedMemory(const StridedMemory<T, Dev>& base, CoordTuple coords) :
    raw_buffer(base.raw_buffer),
    mem_buffer(const_cast<T*>(base.get(get_range_start(coords)))),
    mem_strides(TuplePad<StrideTuple>(base.mem_strides.size()))  // Will be resized.
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
  StridedMemory(T* buffer, ShapeTuple shape, StrideTuple strides) :
    raw_buffer(nullptr),
    mem_buffer(buffer),
    mem_strides(strides)
  {}

  ~StridedMemory() {}

  T* data() H2_NOEXCEPT {
    return mem_buffer;
  }

  const T* data() const H2_NOEXCEPT {
    return mem_buffer;
  }

  const T* const_data() const H2_NOEXCEPT {
    return mem_buffer;
  }

  StrideTuple strides() const H2_NOEXCEPT {
    return mem_strides;
  }

  /** Get the index of coords in the buffer. */
  DataIndexType get_index(const SingleCoordTuple& coords) const H2_NOEXCEPT {
    return inner_product<DataIndexType>(coords, mem_strides);
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
};

}  // namespace h2
