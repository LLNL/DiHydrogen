////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Manages a raw memory buffer.
 */

#include <cstddef>

#include "h2/tensor/tensor_types.hpp"

namespace h2 {

namespace internal {

template <typename T, Device Dev>
struct Allocator {
  static T* allocate(std::size_t size);
  static void deallocate(T* buf);
};

template <typename T>
struct Allocator<T, Device::CPU> {
  static T* allocate(std::size_t size) {
    return new T[size];
  }

  static void deallocate(T* buf) {
    delete[] buf;
  }
};

}  // namespace internal

/**
 * Manage a raw buffer of daaa on a device.
 */
template <typename T, Device Dev>
class RawBuffer {
public:

  RawBuffer() : buffer(nullptr), buffer_size(0) {}
  RawBuffer(std::size_t size) : buffer(nullptr), buffer_size(size) {
    ensure();
  }
  ~RawBuffer() {
    release();
  }

  void ensure() {
    if (!buffer) {
      buffer = internal::Allocator<T, Dev>::allocate(buffer_size);
    }
  }

  void release() {
    if (buffer) {
      internal::Allocator<T, Dev>::deallocate(buffer);
      buffer = nullptr;
    }
  }

  T* data() H2_NOEXCEPT {
    return buffer;
  }

  const T* data() const H2_NOEXCEPT {
    return buffer;
  }

  const T* const_data() const H2_NOEXCEPT {
    return buffer;
  }

  std::size_t size() const H2_NOEXCEPT {
    return buffer_size;
  }

private:
  T* buffer;  /**< Internal buffer. */
  std::size_t buffer_size;   /**< Number of elements in buffer. */
};



}  // namespace h2
