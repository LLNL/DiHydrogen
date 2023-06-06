////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "distconv/dnn_backend/dnn_backend.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/util/util_gpu_dnn.hpp"

#include <numeric>
#include <vector>

namespace distconv
{

template <typename Tensor>
void GPUDNNBackend::setup_filter_descriptor(FilterDescriptor_t& desc,
                                            Tensor const& tensor)
{
    auto const dt = util::get_dnnlib_type<typename Tensor::data_type>();
    auto const shape = tensor.get_local_real_shape().template get_vector<int>();
    set_filter_descriptor(desc, dt, shape.size(), util::reverse(shape).data());
}

template <typename Tensor, typename ShapeType>
void GPUDNNBackend::setup_tensor_descriptor(TensorDescriptor_t& desc,
                                            Tensor const& tensor,
                                            ShapeType const& shape)
{
    auto const dt = util::get_dnnlib_type<typename Tensor::data_type>();
    assert_eq(tensor.get_num_dims(), shape.num_dims());

    if (shape.get_size() == 0)
        return;

    // set descriptor for input tensor
    // The size should include halo regions. Convolution will not be
    // done for the halo regions by disabling padding
    IndexVector strides = tensor::get_strides(
        tensor.get_local_shape(), tensor.get_halo_width(), tensor.get_pitch());

    util::MPIPrintStreamDebug()
        << "setup_tensor_descriptor. "
        << "tensor: " << tensor << ", shape: " << util::join_array(shape, ", ")
        << ", strides: " << util::join_array(strides, ", ") << "\n";

    set_tensor_descriptor(desc,
                          dt,
                          shape.num_dims(),
                          util::reverse(IntVector(shape)).data(),
                          util::reverse(strides).get_vector<int>().data());
}

template <typename Tensor>
void GPUDNNBackend::setup_tensor_descriptor(TensorDescriptor_t& desc,
                                            Tensor const& tensor,
                                            IntVector const& halo_fwd,
                                            IntVector const& halo_bwd)
{
    auto shape = tensor.get_local_shape();
    shape = shape + tensor::Shape(halo_fwd) + tensor::Shape(halo_bwd);
    return setup_tensor_descriptor(desc, tensor, shape);
}

template <typename Tensor>
void GPUDNNBackend::setup_tensor_descriptor(
    TensorDescriptor_t& desc,
    Tensor const& tensor,
    std::vector<bool> const& include_halo_fwd,
    std::vector<bool> const& include_halo_bwd)
{
    int const nd = tensor.get_num_dims();
    auto const overlap = tensor.get_overlap();
    IntVector halo_fwd(nd, 0), halo_bwd(nd, 0);
    for (int i = 0; i < nd; ++i)
    {
        if (include_halo_bwd[i])
            halo_bwd[i] = overlap[i];
        if (include_halo_fwd[i])
            halo_fwd[i] = overlap[i];
    }
    setup_tensor_descriptor(desc, tensor, halo_fwd, halo_bwd);
}

template <typename Tensor>
void GPUDNNBackend::setup_tensor_descriptor(TensorDescriptor_t& desc,
                                            Tensor const& tensor,
                                            bool include_halo)
{
    std::vector<bool> include_halo_array(tensor.get_num_dims(), include_halo);
    setup_tensor_descriptor(
        desc, tensor, include_halo_array, include_halo_array);
}

} // namespace distconv
