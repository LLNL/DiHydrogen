#pragma once

#include "stream.hpp"

#include <cuda_runtime.h>

namespace distconv
{
namespace tensor
{

inline cudaStream_t get_gpu_stream(DefaultStream const& s)
{
  return 0;
}

inline cudaStream_t get_gpu_stream(cudaStream_t const& s)
{
  return s;
}

}  // namespace tensor
}  // namespace distconv
