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
#include "h2/core/types.hpp"
#include "h2/tensor/tensor.hpp"
#include <cstdint>

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

  ComputeStream stream{Dev};

  DeviceBuf<DispatchType, Dev> buf{1};
  buf.fill(static_cast<DispatchType>(false));
  dispatch_test(Dev, buf.buf);
  REQUIRE(read_ele<Dev>(buf.buf, stream) == true);
}

// Dynamic dispatch test:
using dyndist_cust_type_t = std::uint8_t;

namespace h2
{

static_assert(!IsH2ComputeType_v<dyndist_cust_type_t>);
template <>
TypeInfo get_h2_type<dyndist_cust_type_t>()
{
  return TypeInfo::make<dyndist_cust_type_t>(TypeInfo::min_user_token);
}

}

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

void dyndist_cust_test(CPUDev_t,
                       Tensor<dyndist_cust_type_t>& dst,
                       const Tensor<dyndist_cust_type_t>& src)
{
  for (DataIndexType i = 0; i < src.numel(); ++i)
  {
    dst.data()[i] += src.const_data()[i] + 1;
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

// This exists but is not used if we do not test with GPU support.
[[maybe_unused]] void dyndist_cust_test(GPUDev_t,
                                        Tensor<dyndist_cust_type_t>& dst,
                                        const Tensor<dyndist_cust_type_t>& src)
{
  for (DataIndexType i = 0; i < src.numel(); ++i)
  {
    write_ele<Device::GPU>(
        dst.data(),
        i,
        static_cast<dyndist_cust_type_t>(
            read_ele<Device::GPU>(dst.data(), i, dst.get_stream())
            + read_ele<Device::GPU>(src.const_data(), i, src.get_stream()) + 1),
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

  H2_DEVICE_DISPATCH(src.get_device(),
                     do_dispatch(_dispatch_table_dyndist_tester_cpu,
                                 "dyndist_tester_cpu",
                                 DispatchOn<2>(dst, src),
                                 CPUDev_t{},
                                 dst,
                                 src),
                     do_dispatch(_dispatch_table_dyndist_tester_gpu,
                                 "dyndist_tester_gpu",
                                 DispatchOn<2>(dst, src),
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

TEMPLATE_LIST_TEST_CASE("Dynamic dispatch to non-native compute type works",
                        "[dispatch]",
                        AllDevList)
{
  constexpr Device Dev = TestType::value;
  using TensorType = Tensor<dyndist_cust_type_t>;
  constexpr dyndist_cust_type_t src_val = static_cast<dyndist_cust_type_t>(20);
  constexpr dyndist_cust_type_t dst_val = static_cast<dyndist_cust_type_t>(22);
  constexpr dyndist_cust_type_t final_dst_val =
      static_cast<dyndist_cust_type_t>(43);

  // Register for dispatch:
  dispatch_register("dyndist_tester_cpu",
                    get_dispatch_key(get_h2_type<dyndist_cust_type_t>(),
                                     get_h2_type<dyndist_cust_type_t>()),
                    static_cast<void (*)(CPUDev_t,
                                         Tensor<dyndist_cust_type_t>&,
                                         const Tensor<dyndist_cust_type_t>&)>(
                        dyndist_cust_test));
#ifdef H2_TEST_WITH_GPU
  dispatch_register("dyndist_tester_gpu",
                    get_dispatch_key(get_h2_type<dyndist_cust_type_t>(),
                                     get_h2_type<dyndist_cust_type_t>()),
                    static_cast<void (*)(GPUDev_t,
                                         Tensor<dyndist_cust_type_t>&,
                                         const Tensor<dyndist_cust_type_t>&)>(
                        dyndist_cust_test));
#endif

  TensorType src_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};
  TensorType dst_tensor{Dev, {4, 6}, {DT::Sample, DT::Any}};

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

  // Unregister from dispatch:
  dispatch_unregister("dyndist_tester_cpu",
                      get_dispatch_key(get_h2_type<dyndist_cust_type_t>(),
                                       get_h2_type<dyndist_cust_type_t>()));
#ifdef H2_TEST_WITH_GPU
  dispatch_unregister("dyndist_tester_gpu",
                      get_dispatch_key(get_h2_type<dyndist_cust_type_t>(),
                                       get_h2_type<dyndist_cust_type_t>()));
#endif
}
