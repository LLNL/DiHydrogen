////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
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
#include "h2/tensor/tensor_utils.hpp"
#include "h2/tensor/raw_buffer.hpp"
#include "h2/utils/typename.hpp"

namespace h2 {

/**
 * Return the strides needed for a tensor of the given shape to be
 * contiguous.
 */
constexpr inline StrideTuple
get_contiguous_strides(const ShapeTuple& shape) H2_NOEXCEPT
{
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
  const ShapeTuple& shape,
  const StrideTuple& strides) H2_NOEXCEPT {
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
 * Return the extent of a buffer implied by a shape and strides.
 *
 * This is in elements, not bytes.
 */
constexpr inline std::size_t
get_extent_from_strides(const ShapeTuple& shape,
                        const StrideTuple& strides) H2_NOEXCEPT
{
  if (shape.is_empty())
  {
    return 0;
  }
  // Implementation note:
  // Think of this as getting the offset of the last index in shape,
  // then adding 1 to account for the last element.
  return inner_product<std::size_t>(
             map(shape, [](ShapeTuple::type x) { return x - 1; }), strides)
         + 1;
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
    StridedMemory(bool lazy = false,
                  const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : raw_buffer(nullptr),
      mem_offset(INVALID_OFFSET),
      mem_strides{},
      mem_shape{},
      stream(stream_),
      is_mem_lazy(lazy)
  {}

  /** Allocate memory for shape, with unit strides. */
  StridedMemory(const ShapeTuple& shape, bool lazy = false,
                const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : StridedMemory(shape, get_contiguous_strides(shape), lazy, stream_)
  {}

  /** Allocate memory for shape with the given strides. */
  StridedMemory(const ShapeTuple& shape,
                const StrideTuple& strides,
                bool lazy = false,
                const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
      : StridedMemory(lazy, stream_)
  {
    H2_ASSERT_DEBUG(shape.size() == strides.size(),
                    "Shape and strides must be the same size");
    if (!shape.is_empty())
    {
      H2_ASSERT_DEBUG(
        get_extent_from_strides(shape, strides) >= product<std::size_t>(shape),
        "Provided strides are not sane");
      mem_strides = strides;
      mem_shape = shape;
      make_raw_buffer(lazy);
      mem_offset = 0;
    }
  }

  /** View a subregion of an existing memory region. */
  StridedMemory(const StridedMemory<T, Dev>& base,
                const IndexRangeTuple& coords)
      : raw_buffer(base.raw_buffer),
        mem_offset(INVALID_OFFSET),
        // mem_shape and mem_strides are set below.
        stream(base.stream),
        is_mem_lazy(base.is_lazy())
  {
    H2_ASSERT_DEBUG(coords.size() <= base.mem_strides.size(),
                    "coords size not compatible with strides");
    if (is_index_range_empty(coords))
    {
      mem_strides = StrideTuple{};
      mem_shape = ShapeTuple{};
    }
    else
    {
      mem_offset = base.get_index(get_index_range_start(coords));
      mem_shape = get_index_range_shape(coords, base.shape());
      if (mem_shape.is_empty())
      {
        // mem_shape is empty if the coordinates are all scalars.
        // Reset to have a shape and stride of 1.
        mem_shape = ShapeTuple(1);
        mem_strides = StrideTuple(1);
      }
      else if (all_of(mem_shape, [](ShapeTuple::type x) { return x == 1; }))
      {
        // mem_shape is all 1s if every coordinate is a range of length
        // 1. Hence we essentially have a scalar and are contiguous,
        // so force the strides to look contiguous.
        mem_strides = StrideTuple(TuplePad<StrideTuple>(mem_shape.size(), 1));
      }
      else
      {
        // Compute strides as normal.
        for (typename StrideTuple::size_type i = 0; i < base.mem_strides.size();
             ++i)
        {
          if (i >= coords.size() || !coords[i].is_scalar())
          {
            mem_strides.append(base.mem_strides[i]);
          }
        }
      }
    }
  }

  /** Wrap an existing memory buffer. */
  StridedMemory(T* buffer,
                const ShapeTuple& shape,
                const StrideTuple& strides,
                const ComputeStream<Dev>& stream_ = ComputeStream<Dev>{})
    : raw_buffer(nullptr),
      mem_offset(0),
      mem_strides(strides),
      mem_shape(shape),
      stream(stream_),
      is_mem_lazy(false)
  {
    H2_ASSERT_DEBUG(buffer
                    || shape.is_empty()
                    || any_of(shape, [](ShapeTuple::type x) { return x == 0; }),
                    "Null buffer but non-zero shape provided to StridedMemory");
    H2_ASSERT_DEBUG(shape.is_empty()
                        || get_extent_from_strides(shape, strides)
                               >= product<std::size_t>(shape),
                    "Provided strides are not sane");
    std::size_t size = get_extent_from_strides(shape, strides);
    raw_buffer = std::make_shared<raw_buffer_t>(buffer, size, stream);
  }

  ~StridedMemory()
  {
    if (raw_buffer)
    {
      raw_buffer->register_release(stream);
    }
  }

  void ensure(bool attempt_recover = true)
  {
    if (raw_buffer)
    {
      if (is_lazy())
      {
        raw_buffer->ensure();
      }
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
      make_raw_buffer(false);
    }
    old_raw_buffer.reset();  // Drop reference to old raw buffer.
  }

  void release()
  {
    if (raw_buffer)
    {
      raw_buffer->register_release(stream);
      old_raw_buffer = raw_buffer;
      raw_buffer.reset();
    }
  }

  /**
   * Return the size of the memory, in bytes.
   *
   * This gives the size of the entire underlying buffer. It may be
   * larger than expected due to views or non-contiguous strides.
   *
   * This is mainly for internal use.
   */
  std::size_t size() const H2_NOEXCEPT
  {
    if (raw_buffer)
    {
      return raw_buffer->size();
    }
    return 0;
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
    if (raw_buffer && !mem_shape.is_empty())
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
  DataIndexType get_index(const ScalarIndexTuple& coords) const H2_NOEXCEPT {
    return inner_product<DataIndexType>(coords, mem_strides);
  }

  /**
   * Return the point corresponding to the given index in the
   * generalized column-major order.
   */
  ScalarIndexTuple get_coord(DataIndexType idx) const H2_NOEXCEPT
  {
    ScalarIndexTuple coord(TuplePad<ScalarIndexTuple>(mem_shape.size()));
    for (typename ShapeTuple::size_type i = 0; i < mem_shape.size(); ++i)
    {
      coord[i] = (idx / mem_strides[i]) % mem_shape[i];
    }
    return coord;
  }

  /** Return a pointer to the memory at the given coordinates. */
  T* get(const ScalarIndexTuple& coords) H2_NOEXCEPT {
    H2_ASSERT_DEBUG(data(), "No memory");
    return &(data()[get_index(coords)]);
  }

  const T* get(const ScalarIndexTuple& coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(data(), "No memory");
    return &(data()[get_index(coords)]);
  }

  const T* const_get(const ScalarIndexTuple& coords) const H2_NOEXCEPT {
    H2_ASSERT_DEBUG(const_data(), "No memory");
    return &(const_data()[get_index(coords)]);
  }

  ComputeStream<Dev> get_stream() const H2_NOEXCEPT
  {
    return stream;
  }

  void set_stream(const ComputeStream<Dev>& stream, bool set_raw = false)
  {
    stream = stream;
    if (raw_buffer && set_raw)
    {
      raw_buffer->set_stream(stream);
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
  /**
   * Reference to the prior `raw_buffer`, which may be used when
   * recovering existing memory from views using `ensure`.
   */
  std::weak_ptr<raw_buffer_t> old_raw_buffer;
  StrideTuple mem_strides;  /**< Strides associated with the memory. */
  ShapeTuple mem_shape;  /**< Shape describing the extent of the memory. */
  ComputeStream<Dev> stream;  /**< Compute stream for operations. */
  bool is_mem_lazy;  /**< Whether allocation is lazy. */

  /** Helper to create a raw buffer if size is non-empty. */
  void make_raw_buffer(bool lazy)
  {
    // Do not allocate a RawBuffer for empty memory.
    if (!mem_shape.is_empty())
    {
      const std::size_t size = get_extent_from_strides(mem_shape, mem_strides);
      if (size) {
        raw_buffer = std::make_shared<raw_buffer_t>(size, lazy, stream);
      }
    }
  }
};

/** Support printing StridedMemory. */
template <typename T, Device Dev>
inline std::ostream& operator<<(std::ostream& os,
                                const StridedMemory<T, Dev>& mem)
{
  os << "StridedMemory<" << TypeName<T>() << ", " << Dev << ">("
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
