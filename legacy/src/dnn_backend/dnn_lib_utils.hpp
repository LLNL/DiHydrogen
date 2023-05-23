////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "distconv/dnn_backend/dnn_backend.hpp"

#include <stdexcept>
#include <variant>

namespace distconv
{

// This is just a quick wrapper around the "alpha"/"beta" scaling
// parameters needed for the cuDNN/MIOpen interface.
struct host_scalar
{
    std::variant<float, double> val;
    explicit host_scalar(float const v) : val{v} {}
    explicit host_scalar(double const v) : val{v} {}
    void const* get() const
    {
        return std::visit([](auto&& x) { return static_cast<void const*>(&x); },
                          val);
    }
    operator void const*() const { return get(); }
}; // host_scalar

inline host_scalar make_host_scalar(GPUDNNBackend::DataType_t const dt,
                                    double const v)
{
#if H2_HAS_CUDA
    switch (dt)
    {
    case CUDNN_DATA_HALF: [[fallthrough]];
    case CUDNN_DATA_FLOAT: return host_scalar{static_cast<float>(v)};
    case CUDNN_DATA_DOUBLE: return host_scalar{v};
    default:
        throw std::runtime_error("Only float, double, and half are supported.");
    }
#elif H2_HAS_ROCM
    switch (dt)
    {
    case miopenHalf: [[fallthrough]];
    case miopenFloat: return host_scalar{static_cast<float>(v)};
    default: throw std::runtime_error("Only float and half are supported.");
    }
#endif
}

// Dims are NC(D)HW (nsamples = dims[0])
inline std::vector<int> get_fully_packed_strides(std::vector<int> const& dims)
{
    size_t const ndims = dims.size();
    std::vector<int> strides(ndims, 1);
    std::partial_sum(dims.rbegin(),
                     std::prev(dims.rend()),
                     std::next(strides.rbegin()),
                     std::multiplies<int>{});
    return strides;
}

inline size_t datatype_size(GPUDNNBackend::DataType_t dt)
{
#if H2_HAS_CUDA
    switch (dt)
    {
    case CUDNN_DATA_FLOAT: return sizeof(float);
    case CUDNN_DATA_DOUBLE: return sizeof(double);
    case CUDNN_DATA_HALF: return sizeof(short);
    default:
        throw std::runtime_error("Only float, double, and half are supported.");
    }
#elif H2_HAS_ROCM
    switch (dt)
    {
    case miopenHalf: return sizeof(short);
    case miopenFloat: return sizeof(float);
    default: throw std::runtime_error("Only float and half are supported.");
    }
#endif
    return 1UL;
}

} // namespace distconv
