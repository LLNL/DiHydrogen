////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"

#include <type_traits>

#include "h2/core/dispatch.hpp"
#include "h2/tensor/base_utils.hpp"
#include "h2/utils/unique_ptr_cast.hpp"
#include "h2/loops/cpu_loops.hpp"

namespace h2
{

namespace internal
{

void copy_same_type(BaseTensor& dst, const BaseTensor& src)
{
  dst.resize(src.shape(), src.dim_types(), src.strides());
  dst.ensure();
  if (src.is_contiguous())
  {
    copy_buffer(dst.storage_data(),
                dst.get_stream(),
                src.const_storage_data(),
                src.get_stream(),
                src.numel() * src.get_type_info().get_size());
  }
  else
  {
    // TODO: We may be able to optimize the non-contiguous case.
    // For now, we just copy the entire buffer.
    copy_buffer(dst.storage_data(),
                dst.get_stream(),
                src.const_storage_data(),
                src.get_stream(),
                get_extent_from_strides(src.shape(), src.strides())
                    * src.get_type_info().get_size());
  }
}

}  // namespace internal

template <typename DstT>
std::unique_ptr<Tensor<DstT>> cast(BaseTensor& src)
{
  // H2_DISPATCH_NAME: cast
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT_CPU: impl::cast_impl("CPUDev_t", "Tensor<DstT>&", "const Tensor<{T1}>&")
  // H2_DISPATCH_INIT_GPU: impl::cast_impl("GPUDev_t", "Tensor<DstT>&", "const Tensor<{T1}>&")

  if (src.get_type_info() == get_h2_type<DstT>())
  {
    auto view = base::view(src);
    return downcast_uptr<Tensor<DstT>>(view);
  }

  auto dst = std::make_unique<Tensor<DstT>>(src.get_device(),
                                            src.shape(),
                                            src.dim_types(),
                                            src.strides(),
                                            StrictAlloc,
                                            src.get_stream());

  // H2_DISPATCH_GET_DEVICE: "src.get_device()"
  // H2_DISPATCH_ON: "src"
  // H2_DISPATCH_ARGS_CPU: "CPUDev_t{}", "*dst", "src"
  // H2_DISPATCH_ARGS_GPU: "GPUDev_t{}", "*dst", "src"
  // H2_DO_DISPATCH

  return dst;
}

#define PROTO(device, t1)                                       \
  template std::unique_ptr<Tensor<t1>> cast<t1>(BaseTensor&)
H2_INSTANTIATE_DEV_1(none)
#undef PROTO

std::unique_ptr<BaseTensor> cast(const TypeInfo& type, BaseTensor& src)
{
  // H2_DISPATCH_NAME: cast
  // H2_DISPATCH_NUM_TYPES: 2
  // H2_DISPATCH_INIT_CPU: impl::cast_impl("CPUDev_t", "Tensor<{T1}>&", "const Tensor<{T2}>&")
  // H2_DISPATCH_INIT_GPU: impl::cast_impl("GPUDev_t", "Tensor<{T1}>&", "const Tensor<{T2}>&")

  if (src.get_type_info() == type)
  {
    return base::view(src);
  }

  auto dst = base::make_tensor(type,
                               src.get_device(),
                               src.shape(),
                               src.dim_types(),
                               src.strides(),
                               StrictAlloc,
                               src.get_stream());

  // H2_DISPATCH_GET_DEVICE: "src.get_device()"
  // H2_DISPATCH_ON: "type", "src"
  // H2_DISPATCH_ARGS_CPU: "CPUDev_t{}", "*dst", "src"
  // H2_DISPATCH_ARGS_GPU: "GPUDev_t{}", "*dst", "src"
  // H2_DO_DISPATCH

  return dst;
};

namespace impl
{

template <typename DstT, typename SrcT>
void cast_impl(CPUDev_t, Tensor<DstT>& dst, const Tensor<SrcT>& src)
{
  static_assert(std::is_convertible_v<SrcT, DstT>,
                "Attempt to cast between inconvertible types");
  const SrcT* __restrict__ src_buf = src.const_data();
  DstT* __restrict__ dst_buf = dst.data();
  if (src.is_contiguous())
  {
    h2::cpu::elementwise_loop(
        [](const SrcT val) -> DstT { return static_cast<DstT>(val); },
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
H2_INSTANTIATE_CPU_2
#undef PROTO

}  // namespace impl

}  // namespace h2
