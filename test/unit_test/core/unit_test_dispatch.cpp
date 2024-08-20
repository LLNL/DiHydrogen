////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include "h2/core/dispatch.hpp"
#include "h2/core/device.hpp"
#include "h2/tensor/tensor.hpp"

#include "../tensor/utils.hpp"

using namespace h2;

TEMPLATE_LIST_TEST_CASE("Static dispatch works for compute types",
                        "[dispatch]",
                        AllDevComputeTypePairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using DispatchType = meta::tlist::At<TestType, 1>;

  DeviceBuf<DispatchType, Dev> buf{1};
  buf.fill(static_cast<DispatchType>(0));
  if constexpr (Dev == Device::CPU)
  {
    dispatch_test(Dev, buf.buf);
    REQUIRE(buf.buf[0] == 42);
  }
#ifdef H2_TEST_WITH_GPU
  else if constexpr (Dev == Device::GPU)
  {
    // Does nothing on GPUs.
    dispatch_test(Dev, buf.buf);
    REQUIRE(read_ele<Device::GPU>(buf.buf, 0) == 42);
  }
#endif
}

template <>
void h2::impl::dispatch_test_impl<bool>(CPUDev_t, bool* v)
{
  *v = true;
}

// This is HAS_GPU rather than TEST_WITH_GPU because dispatch_test will
// always generate dispatch code for the GPU version whenever H2 has
// GPU support.
#ifdef H2_HAS_GPU
template <>
void h2::impl::dispatch_test_impl<bool>(GPUDev_t, bool* v)
{
  write_ele<Device::GPU, bool>(v, 0, true, 0);
}
#endif

TEMPLATE_LIST_TEST_CASE("Static dispatch works for new types",
                        "[dispatch]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using DispatchType = bool;

  DeviceBuf<DispatchType, Dev> buf{1};
  buf.fill(static_cast<DispatchType>(false));
  dispatch_test(Dev, buf.buf);
  REQUIRE(read_ele<Dev>(buf.buf, 0) == true);
}

// Dynamic dispatch test:

namespace
{

template <typename T1, typename T2>
void dyndist_test(CPUDev_t, Tensor<T1>& dst, const Tensor<T2>& src)
{
  for (DataIndexType i = 0; i < src.numel(); ++i)
  {
    dst.data()[i] += src.const_data()[i];
  }
}

#ifdef H2_HAS_GPU
template <typename T1, typename T2>
void dyndist_test(GPUDev_t, Tensor<T1>& dst, const Tensor<T2>& src)
{
  for (DataIndexType i = 0; i < src.numel(); ++i)
  {
    write_ele<Device::GPU>(
        dst.data(),
        i,
        static_cast<T1>(
            read_ele<Device::GPU>(dst.data(), i, dst.get_stream())
            + read_ele<Device::GPU>(src.const_data(), i, src.get_stream())),
        dst.get_stream());
  }
}
#endif

void dyndist_tester(BaseTensor& dst, const BaseTensor& src)
{
  // These dispatch tables are manually generated:
  static std::array<h2::internal::DispatchFunctionEntry, 16>
      _dispatch_table_dyndist_tester_cpu = {{
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<float>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<float>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<float>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<float>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<float>&, const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<float>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<float>&, const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<float>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<double>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<double>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<double>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<double>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<double>&, const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<double>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<double>&, const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<double>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<std::int32_t>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<std::int32_t>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<std::int32_t>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<std::int32_t>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(CPUDev_t,
                                    Tensor<std::int32_t>&,
                                    const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<std::int32_t>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(CPUDev_t,
                                    Tensor<std::int32_t>&,
                                    const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<std::int32_t>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<std::uint32_t>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<std::uint32_t>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   CPUDev_t, Tensor<std::uint32_t>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              CPUDev_t,
                                              Tensor<std::uint32_t>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(CPUDev_t,
                                    Tensor<std::uint32_t>&,
                                    const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<std::uint32_t>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(CPUDev_t,
                                    Tensor<std::uint32_t>&,
                                    const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               CPUDev_t,
               Tensor<std::uint32_t>&,
               const Tensor<std::uint32_t>&>::call},
      }};
#ifdef H2_HAS_GPU
  static std::array<h2::internal::DispatchFunctionEntry, 16>
      _dispatch_table_dyndist_tester_gpu = {{
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<float>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<float>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<float>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<float>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<float>&, const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<float>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<float>&, const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<float>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<double>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<double>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<double>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<double>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<double>&, const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<double>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<double>&, const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<double>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<std::int32_t>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<std::int32_t>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<std::int32_t>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<std::int32_t>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(GPUDev_t,
                                    Tensor<std::int32_t>&,
                                    const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<std::int32_t>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(GPUDev_t,
                                    Tensor<std::int32_t>&,
                                    const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<std::int32_t>&,
               const Tensor<std::uint32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<std::uint32_t>&, const Tensor<float>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<std::uint32_t>&,
                                              const Tensor<float>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(
                   GPUDev_t, Tensor<std::uint32_t>&, const Tensor<double>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<void,
                                              GPUDev_t,
                                              Tensor<std::uint32_t>&,
                                              const Tensor<double>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(GPUDev_t,
                                    Tensor<std::uint32_t>&,
                                    const Tensor<std::int32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<std::uint32_t>&,
               const Tensor<std::int32_t>&>::call},
          {reinterpret_cast<void*>(
               static_cast<void (*)(GPUDev_t,
                                    Tensor<std::uint32_t>&,
                                    const Tensor<std::uint32_t>&)>(
                   dyndist_test)),
           &h2::internal::DispatchFunctionWrapper<
               void,
               GPUDev_t,
               Tensor<std::uint32_t>&,
               const Tensor<std::uint32_t>&>::call},
      }};
#endif

  if (!h2::internal::all_h2_compute_types(dst, src))
  {
    throw H2FatalException("Attempt to dispatch on non-H2 compute type");
  }
  const auto dispatch_key = h2::internal::get_dispatch_key(dst, src);
  H2_ASSERT_DEBUG(dispatch_key < _dispatch_table_dyndist_tester_cpu.size(),
                  "Bad dispatch key");
  H2_DEVICE_DISPATCH(src.get_device(),
                     h2::internal::dispatch_call(
                         _dispatch_table_dyndist_tester_cpu[dispatch_key],
                         CPUDev_t{},
                         dst,
                         src),
                     h2::internal::dispatch_call(
                         _dispatch_table_dyndist_tester_gpu[dispatch_key],
                         GPUDev_t{},
                         dst,
                         src));
}

}  // anonymous namespace

TEMPLATE_LIST_TEST_CASE("Dynamic dispatch works for H2 compute types",
                        "[dispatch]",
                        AllDevComputeTypePairsPairsList)
{
  constexpr Device Dev = meta::tlist::At<TestType, 0>::value;
  using SrcType = meta::tlist::At<meta::tlist::At<TestType, 1>, 0>;
  using DstType = meta::tlist::At<meta::tlist::At<TestType, 1>, 1>;
  using SrcTensorType = Tensor<SrcType>;
  using DstTensorType = Tensor<DstType>;
  constexpr SrcType src_val = static_cast<SrcType>(20);
  constexpr DstType dst_val = static_cast<DstType>(22);
  constexpr DstType final_dst_val = static_cast<DstType>(42);

  SrcTensorType src_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};
  DstTensorType dst_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    write_ele<Dev>(src_tensor.data(), i, src_val, src_tensor.get_stream());
    write_ele<Dev>(dst_tensor.data(), i, dst_val, dst_tensor.get_stream());
  }

  REQUIRE_NOTHROW(dyndist_tester(dst_tensor, src_tensor));

  for (DataIndexType i = 0; i < src_tensor.numel(); ++i)
  {
    REQUIRE(read_ele<Dev>(src_tensor.data(), i, src_tensor.get_stream())
            == src_val);
    REQUIRE(read_ele<Dev>(dst_tensor.data(), i, dst_tensor.get_stream())
            == final_dst_val);
  }
}
