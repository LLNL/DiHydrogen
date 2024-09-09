////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "h2/tensor/init/fill.hpp"

#include "h2/core/dispatch.hpp"
#include "h2/utils/typename.hpp"
#include "h2/loops/cpu_loops.hpp"


namespace h2
{

void zero(BaseTensor& tensor)
{
  // H2_DISPATCH_NAME: zero
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT: zero<{T1}>("Tensor<{T1}>&")

  if (tensor.is_contiguous())
  {
    // H2_DISPATCH_ON: "tensor"
    // H2_DISPATCH_ARGS: "tensor"
    // H2_DO_DISPATCH
  }
  else
  {
    throw H2FatalException("Zero not implemented for non-contiguous tensors");
  }
}

namespace impl
{

template <typename T>
void fill_impl(CPUDev_t, Tensor<T>& tensor, const T& val)
{
  if (tensor.is_empty())
  {
    return;
  }
  if (tensor.is_contiguous())
  {
    cpu::elementwise_loop(
        [&val]() -> T { return val; }, tensor.numel(), tensor.data());
  }
  else
  {
    throw H2FatalException("Not supporting non-contiguous tensors");
  }
}

#define PROTO(device, t1)                                       \
  template void fill_impl<t1>(device, Tensor<t1>&, const t1&)
H2_INSTANTIATE_CPU_1
#undef PROTO

}  // namespace impl

template <typename T>
void fill(BaseTensor& tensor, const T& val)
{
  // H2_DISPATCH_NAME: fill
  // H2_DISPATCH_NUM_TYPES: 1
  // H2_DISPATCH_INIT_CPU: impl::fill_impl("CPUDev_t", "Tensor<{T1}>&", "const {T1}&")
  //  H2_DISPATCH_INIT_GPU: impl::fill_impl("GPUDev_t", "Tensor<{T1}>&", "const {T1}&")

  // Ensure val matches the tensor's type.
  // TODO: Relax this to allow conversion.
  if (get_h2_type<T>() != tensor.get_type_info())
  {
    throw H2FatalException("Cannot fill a tensor of ",
                           tensor.get_type_info(),
                           " with ",
                           val,
                           " of type ",
                           TypeName<T>());
  }

  // H2_DISPATCH_GET_DEVICE: "tensor.get_device()"
  // H2_DISPATCH_ON: "tensor"
  // H2_DISPATCH_ARGS_CPU: "CPUDev_t{}", "tensor", "val"
  //  H2_DISPATCH_ARGS_GPU: "GPUDev_t{}", "tensor", "val"
  // H2_DO_DISPATCH
}

#define PROTO(device, t1) template void fill<t1>(BaseTensor&, const t1&)
H2_INSTANTIATE_DEV_1(none)
#undef PROTO

}  // namespace h2
