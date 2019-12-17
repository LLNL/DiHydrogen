#pragma once

namespace distconv {
namespace tensor {

inline cudaStream_t get_cuda_stream(const DefaultStream &s) {
  return 0;
}

inline cudaStream_t get_cuda_stream(const cudaStream_t &s) {
  return s;
}

} // namespace tensor
} // namespace distconv
