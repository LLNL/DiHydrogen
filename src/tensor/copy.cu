////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"

#include "h2/core/dispatch.hpp"
#include "h2/loops/gpu_loops.cuh"

namespace h2
{

namespace impl
{

template <typename DstT, typename SrcT>
void cast_impl(GPUDev_t, Tensor<DstT>& dst, const Tensor<SrcT>& src)
{
  static_assert(std::is_convertible_v<SrcT, DstT>,
                "Attempt to cast between inconvertible types");
  const SrcT* __restrict__ src_buf = src.const_data();
  DstT* __restrict__ dst_buf = dst.data();
  auto stream = create_multi_sync(dst.get_stream(), src.get_stream());
  if (src.is_contiguous())
  {
    h2::gpu::launch_elementwise_loop(
        [] H2_GPU_LAMBDA(const SrcT val) -> DstT { return static_cast<DstT>(val); },
        stream,
        dst.numel(),
        dst_buf,
        src_buf);
  }
  else
  {
    throw H2FatalException("Not currently supporting non-contiguous cast");
  }
}

#define PROTO(device, t1, t2)                                                  \
  template void cast_impl<t1, t2>(device, Tensor<t1>&, const Tensor<t2>&)
H2_INSTANTIATE_GPU_2
#undef PROTO

}  // namespace impl

}  // namespace h2
