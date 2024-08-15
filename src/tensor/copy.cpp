////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"

#include <type_traits>

#include "h2/core/dispatch.hpp"

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

namespace impl
{

template <typename DstT, typename SrcT>
void cast_impl(CPUDev_t, Tensor<DstT>& dst, const Tensor<SrcT>& src)
{
  static_assert(std::is_convertible_v<SrcT, DstT>,
                "Attempt to cast between inconvertible types");
  const SrcT* src_buf = src.const_data();
  DstT* dst_buf = dst.data();
  if (src.is_contiguous())
  {
    for (DataIndexType i = 0; i < product<DataIndexType>(src.shape()); ++i)
    {
      dst_buf[i] = static_cast<DstT>(src_buf[i]);
    }
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
