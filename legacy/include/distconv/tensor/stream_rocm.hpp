#pragma once

#include <hip/hip_runtime.h>

namespace distconv
{
namespace tensor
{

inline hipStream_t get_gpu_stream(DefaultStream const& s)
{
    return 0;
}

inline hipStream_t get_gpu_stream(hipStream_t const& s)
{
    return s;
}

} // namespace tensor
} // namespace distconv
