////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "h2/tensor/tensor.hpp"

#include <El.hpp>

#include <type_traits>

namespace h2
{

/** @brief Metafunction to convert h2::Device to hydrogen::Device. */
template <Device D>
struct HydrogenDeviceT;

template <>
struct HydrogenDeviceT<Device::CPU>
    : std::integral_constant<hydrogen::Device, hydrogen::Device::CPU>
{};

#ifdef H2_HAS_GPU
template <>
struct HydrogenDeviceT<Device::GPU>
    : std::integral_constant<hydrogen::Device, hydrogen::Device::GPU>
{};
#endif

template <Device D>
inline constexpr auto HydrogenDevice = HydrogenDeviceT<D>::value;

/** @brief Metafunction to convert hydrogen::Device to h2::Device. */
template <hydrogen::Device D>
struct H2DeviceT;

template <>
struct H2DeviceT<hydrogen::Device::CPU>
    : std::integral_constant<Device, Device::CPU>
{};

#ifdef H2_HAS_GPU
template <>
struct H2DeviceT<hydrogen::Device::GPU>
    : std::integral_constant<Device, Device::GPU>
{};
#endif

template <Device D>
inline constexpr auto H2Device = H2DeviceT<D>::value;

template <typename T>
bool is_chw_packed(Tensor<T> const& tensor)
{
    if (tensor.is_empty())
        return true;
    if (tensor.stride(0) != decltype(tensor.stride(0)){1})
        return false;

    // Recall that H2 doesn't allow "overlapped" tensors. So the
    // definition of CHW-packed is simply that stride(i-1)*dim(i-1) ==
    // stride(i) for i in the CHW index range.
    auto const ndim = tensor.ndim();
    for (auto i = decltype(ndim){1}; i < ndim - 1; ++i)
        if (tensor.stride(i - 1) * tensor.shape(i - 1) != tensor.stride(i))
            return false;
    return true;
}

template <typename T, hydrogen::Device D>
hydrogen::SyncInfo<D> get_sync_info(El::Matrix<T, D> const& m)
{
    return El::SyncInfoFromMatrix(m);
}

} // namespace h2
