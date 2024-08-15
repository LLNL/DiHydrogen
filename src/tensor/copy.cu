////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"

#include "h2/core/dispatch.hpp"

namespace h2
{

namespace impl
{

template <typename DstT, typename SrcT>
__global__ void
cast_kernel_contiguous(DstT* dst, const SrcT* src, std::size_t len)
{
  const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < len)
  {
    dst[idx] = static_cast<DstT>(src[idx]);
  }
}

template <typename DstT, typename SrcT>
void cast_impl(GPUDev_t, Tensor<DstT>& dst, const Tensor<SrcT>& src)
{
  static_assert(std::is_convertible_v<SrcT, DstT>,
                "Attempt to cast between inconvertible types");
  const SrcT* src_buf = src.const_data();
  DstT* dst_buf = dst.data();
  auto stream = create_multi_sync(dst.get_stream(), src.get_stream());
  if (src.is_contiguous())
  {
    // TODO: Do not hardcode these.
    // TODO: Switch to using a general unary kernel when we have one.
    constexpr std::size_t threads_per_block = 256ull;
    const auto blocks = (dst.numel() + threads_per_block - 1) / threads_per_block;
    gpu::launch_kernel(cast_kernel_contiguous<DstT, SrcT>,
                       blocks,
                       threads_per_block,
                       0,
                       stream.template get_stream<Device::GPU>(),
                       dst.data(),
                       src.const_data(),
                       dst.numel());
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
