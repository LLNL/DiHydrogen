#pragma once

#include <hip/hip_runtime.h>

namespace distconv {
namespace tensor {

inline hipStream_t get_hip_stream(const DefaultStream &s) {
  return 0;
}

inline hipStream_t get_hip_stream(const hipStream_t &s) {
  return s;
}

} // namespace tensor
} // namespace distconv
