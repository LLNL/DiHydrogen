////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <map>
#include <tuple>

#include "dnn_backend.hpp"

namespace distconv
{
/** @brief Type of JIT-compiled convolution **/
enum ConvType
{
    FORWARD = 0,
    BACKWARD_FILTER,
    BACKWARD_DATA
};

/** @brief Convolution parameters **/
struct ConvParams
{
    int pads[3];
    int strides[3];
    int dilation[3];
    int groups;
};

// 5D shape
using s5d = std::tuple<int, int, int, int, int>;
inline void set_value(s5d& shape, int d, int n)
{
    switch (d)
    {
    case 0: std::get<0>(shape) = n; break;
    case 1: std::get<1>(shape) = n; break;
    case 2: std::get<2>(shape) = n; break;
    case 3: std::get<3>(shape) = n; break;
    case 4: std::get<4>(shape) = n; break;
    default: break;
    }
}

/**
 * @brief A convolution descriptor with a one-to-one correspondence to a
 * JIT-compiled library.
 **/
struct ConvDescriptor
{
    // Tensor parameters
    s5d x_shape, x_strides;
    s5d w_shape;
    s5d y_shape, y_strides;

    // Convolution parameters
    ConvType type;
    ConvParams params;

    /**
     * Returns tensor dimensionality from convolution parameters.
     **/
    int get_dimensionality() const
    {
        if (std::get<4>(w_shape) == 0)
        {
            if (std::get<3>(w_shape) == 0)
                return 1;
            return 2;
        }
        return 3;
    }

    std::string hash(bool dynamic_minibatch_size = false) const;
};

// Definition of dace types
typedef void* dacehandle_t;
typedef bool (*dace_setstream_t)(dacehandle_t handle, void const* stream);
typedef void (*dace_exitfunc_t)(dacehandle_t handle);

// void const* below act as catch-all pointers for data types and fwd/bwd
typedef void (*daceprogram_t)(dacehandle_t handle,
                              void const*,
                              void const*,
                              void const*,
                              float alpha,
                              float beta);
typedef size_t (*getworkspacesize_t)(dacehandle_t handle);
typedef void (*setworkspace_t)(dacehandle_t handle, void* workspace);

// Versions of the above JIT-compiled functions with runtime minibatch size
typedef void (*dynbatch_daceprogram_t)(dacehandle_t handle,
                                       void const*,
                                       void const*,
                                       void const*,
                                       int B,
                                       float alpha,
                                       float beta);
typedef size_t (*dynbatch_getworkspacesize_t)(dacehandle_t handle, int B);
typedef void (*dynbatch_setworkspace_t)(dacehandle_t handle,
                                        void* workspace,
                                        int B);

/**
 * @brief An internal DaCe-compiled function's state; contains handles and
 * function pointers and handles.
 **/
struct dace_state
{
    void* library;
    dacehandle_t handle;
    dace_setstream_t setstream_func;
    daceprogram_t func;
    dynbatch_daceprogram_t dynbatch_func;
    dace_exitfunc_t dtor;
    getworkspacesize_t get_ws_size;
    dynbatch_getworkspacesize_t dynbatch_get_ws_size;
    setworkspace_t set_workspace;
    dynbatch_setworkspace_t dynbatch_set_workspace;

    dace_state()
        : library(nullptr),
          handle(nullptr),
          setstream_func(nullptr),
          func(nullptr),
          dynbatch_func(nullptr),
          dtor(nullptr),
          get_ws_size(nullptr),
          dynbatch_get_ws_size(nullptr),
          set_workspace(nullptr),
          dynbatch_set_workspace(nullptr)
    {}
};

template <typename VendorBackendT>
class DaCeDNNBackend : public DNNBackend<VendorBackendT>
{
public:
    using VendorBackend = VendorBackendT;
    using ActivationDescriptor_t =
        typename VendorBackend::ActivationDescriptor_t;
    using ConvBwdDataAlgo_t = typename VendorBackend::ConvBwdDataAlgo_t;
    using ConvBwdFilterAlgo_t = typename VendorBackend::ConvBwdFilterAlgo_t;
    using ConvFwdAlgo_t = typename VendorBackend::ConvFwdAlgo_t;
    using ConvolutionDescriptor_t =
        typename VendorBackend::ConvolutionDescriptor_t;
    using ConvolutionMode_t = typename VendorBackend::ConvolutionMode_t;
    using DataType_t = typename VendorBackend::DataType_t;
    using Event_t = typename VendorBackend::Event_t;
    using FilterDescriptor_t = typename VendorBackend::FilterDescriptor_t;
    using Handle_t = typename VendorBackend::Handle_t;
    using PoolingDescriptor_t = typename VendorBackend::PoolingDescriptor_t;
    using PoolingMode_t = typename VendorBackend::PoolingMode_t;
    using Stream_t = typename VendorBackend::Stream_t;
    using TensorDescriptor_t = typename VendorBackend::TensorDescriptor_t;

    DaCeDNNBackend(MPI_Comm comm, Handle_t handle, Options opts = Options{});
    DaCeDNNBackend(MPI_Comm comm,
                   Handle_t handle,
                   Stream_t stream,
                   Options opts = Options{});
    virtual ~DaCeDNNBackend();

    // Use the other overloads from the base class
    using DNNBackend<VendorBackendT>::convolution_forward;
    using DNNBackend<VendorBackendT>::convolution_bwd_data;
    using DNNBackend<VendorBackendT>::convolution_bwd_filter;

    void convolution_forward(double alpha,
                             TensorDescriptor_t const& xdesc,
                             void const* x,
                             FilterDescriptor_t const& filter_desc,
                             void const* filter_data,
                             ConvolutionDescriptor_t const& conv_desc,
                             ConvFwdAlgo_t const& conv_algo,
                             void* workspace,
                             size_t workspace_bytes,
                             double beta,
                             TensorDescriptor_t const& ydesc,
                             void* y,
                             Stream_t s) const override;

    void convolution_bwd_data(double alpha,
                              FilterDescriptor_t const& filter_desc,
                              void const* filter_data,
                              TensorDescriptor_t const& dy_desc,
                              void const* dy_data,
                              ConvolutionDescriptor_t const& conv_desc,
                              ConvBwdDataAlgo_t const& conv_algo,
                              void* workspace,
                              size_t workspace_bytes,
                              double beta,
                              TensorDescriptor_t const& dx_desc,
                              void* dx_data,
                              Stream_t s) const override;

    void convolution_bwd_filter(double alpha,
                                TensorDescriptor_t const& in_desc,
                                void const* in_data,
                                TensorDescriptor_t const& dy_desc,
                                void const* dy_data,
                                ConvolutionDescriptor_t const& conv_desc,
                                ConvBwdFilterAlgo_t const& conv_algo,
                                void* workspace,
                                size_t workspace_bytes,
                                double beta,
                                FilterDescriptor_t const& dw_desc,
                                void* dw_data,
                                Stream_t s) const override;

    size_t
    get_conv_forward_workspace_size(TensorDescriptor_t const& in_desc,
                                    FilterDescriptor_t const& filter_desc,
                                    ConvolutionDescriptor_t const& conv_desc,
                                    TensorDescriptor_t const& out_desc,
                                    ConvFwdAlgo_t const& algo) const override;

    size_t get_conv_bwd_data_workspace_size(
        FilterDescriptor_t const& filter_desc,
        TensorDescriptor_t const& dy_desc,
        ConvolutionDescriptor_t const& conv_desc,
        TensorDescriptor_t const& dx_desc,
        ConvBwdDataAlgo_t const& algo) const override;

    size_t get_conv_bwd_filter_workspace_size(
        TensorDescriptor_t const& in_desc,
        TensorDescriptor_t const& dy_Desc,
        ConvolutionDescriptor_t const& conv_Desc,
        FilterDescriptor_t const& dw_desc,
        ConvBwdFilterAlgo_t const& algo) const override;

    auto get_fwd_algorithm(std::string const& name,
                           TensorDescriptor_t const& input_desc,
                           void const* input,
                           FilterDescriptor_t const& filter_desc,
                           void const* filter,
                           ConvolutionDescriptor_t const& conv_desc,
                           TensorDescriptor_t const& output_desc,
                           void* output,
                           size_t ws_size) const -> ConvFwdAlgo_t override;

    auto get_bwd_data_algorithm(std::string const& name,
                                FilterDescriptor_t const& filter_desc,
                                void const* filter,
                                TensorDescriptor_t const& d_output_desc,
                                void const* d_output,
                                ConvolutionDescriptor_t const& conv_desc,
                                TensorDescriptor_t const& d_input_desc,
                                void* d_input,
                                size_t ws_size) const
        -> ConvBwdDataAlgo_t override;
    

    auto get_bwd_filter_algorithm(std::string const& name,
                                  TensorDescriptor_t const& input_desc,
                                  void const* input,
                                  TensorDescriptor_t const& d_output_desc,
                                  void const* d_output,
                                  ConvolutionDescriptor_t const& conv_desc,
                                  FilterDescriptor_t const& d_filter_desc,
                                  void* d_filter,
                                  size_t ws_size) const
        -> ConvBwdFilterAlgo_t override;

protected:
    // JIT-compiled libraries
    mutable std::map<ConvDescriptor, dace_state> m_dace_libraries;

    bool descriptor_from_tensors(TensorDescriptor_t const& xdesc,
                                 FilterDescriptor_t const& filter_desc,
                                 ConvolutionDescriptor_t const& conv_desc,
                                 TensorDescriptor_t const& ydesc,
                                 ConvDescriptor& result) const;

    dace_state try_load(const std::string& hash,
                        bool dynamic_minibatch_size) const;

    bool unload(dace_state library);

    bool load_library_or_fallback(const ConvDescriptor& desc,
                                  dace_state& library) const;

    bool invoke(const ConvDescriptor& desc,
                void const*,
                void const*,
                void const*,
                float alpha,
                float beta,
                void* workspace,
                Stream_t stream) const;

}; // class DaCeDNNBackend

} // namespace distconv
