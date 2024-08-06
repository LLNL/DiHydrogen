////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/copy.hpp"


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

}  // namespace h2
