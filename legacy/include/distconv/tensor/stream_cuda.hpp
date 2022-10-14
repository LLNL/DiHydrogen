#pragma once

#include "stream.hpp"

#include <cuda_runtime.h>

namespace distconv {
namespace tensor {

inline cudaStream_t get_gpu_stream(const DefaultStream &s) {
  return 0;
}

inline cudaStream_t get_gpu_stream(const cudaStream_t &s) {
  return s;
}

} // namespace tensor
} // namespace distconv
