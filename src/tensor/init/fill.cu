////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/core/dispatch.hpp"
#include "h2/gpu/runtime.hpp"
#include "h2/loops/gpu_loops.cuh"
#include "h2/tensor/init/fill.hpp"

namespace h2
{

namespace impl
{

template <typename T>
void fill_impl(GPUDev_t, Tensor<T>& tensor, T const& val)
{
  if (tensor.is_empty())
  {
    return;
  }
  if (tensor.is_contiguous())
  {
    T* __restrict__ out = tensor.data();
    h2::gpu::launch_elementwise_loop_with_immediate(
      [] H2_GPU_LAMBDA(T const val_) -> T { return val_; },
      tensor.get_stream(),
      tensor.numel(),
      val,
      out);
  }
  else
  {
    throw H2FatalException("Not supporting non-contiguous tensors");
  }
}

#define PROTO(device, t1)                                                      \
  template void fill_impl<t1>(device, Tensor<t1>&, const t1&);
H2_INSTANTIATE_GPU_1
#undef PROTO

}  // namespace impl

}  // namespace h2
