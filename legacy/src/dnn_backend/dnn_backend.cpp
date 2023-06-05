////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/dnn_backend.hpp"

#include "./dnn_lib_utils.hpp"
#include "distconv/dnn_backend/dnn_backend_impl.hpp"
#include "distconv/dnn_backend/pack_unpack.hpp"
#include "h2/gpu/runtime.hpp"

#include <functional> // std::multiplies
#include <memory>     // std::make_shared
#include <numeric>    // std::exclusive_scan
#include <ostream>    // std::ostream
#include <sstream>    // std::ostringstream
#include <stdexcept>  // std::runtime_error
#include <string>     // std::string
#include <utility>    // std::move
#include <vector>     // std::vector

#define DISTCONV_ASSERT_PTR(ptr)                                               \
    do                                                                         \
    {                                                                          \
        if (!(ptr))                                                            \
        {                                                                      \
            ::distconv::util::PrintStreamError()                               \
                << "Error at " << __FILE__ << ":" << __LINE__ << ": " << #ptr  \
                << " is a null pointer." << std::endl;                         \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define DISTCONV_ASSERT(cond)

#ifdef H2_DEBUG
#define DISTCONV_ASSERT_DEBUG(cond) DISTCONV_ASSERT(conv)
#else
#define DISTCONV_ASSERT_DEBUG(cond)
#endif

#if H2_HAS_ROCM
#include "dnn_backend_miopen.cpp.inc"
#elif H2_HAS_CUDA
#include "dnn_backend_cudnn.cpp.inc"
#endif

namespace distconv
{

template <typename VendorBackendT>
DNNBackend<VendorBackendT>::DNNBackend(MPI_Comm comm,
                                       Handle_t handle,
                                       Options opts)
    : m_opts{std::move(opts)},
      m_handle{handle},
      m_stream_mgr{8},
      m_comms{comm, m_stream_mgr}
{}

template <typename VendorBackendT>
DNNBackend<VendorBackendT>::DNNBackend(MPI_Comm comm,
                                       Handle_t handle,
                                       Stream_t stream,
                                       Options opts)
    : m_opts{std::move(opts)},
      m_handle{handle},
      m_stream_mgr{8, stream},
      m_comms{comm, m_stream_mgr}
{}

template <typename VendorBackendT>
std::string DNNBackend<VendorBackendT>::get_name() const
{
    return std::string("DNNBackend<") + VendorBackendT::get_name() + ">";
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_comm() const noexcept -> MPI_Comm
{
    return m_comms.get_comm();
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_al_comm() -> AlCommType&
{
    return m_comms.get_al_nccl_comm();
}

template <typename VendorBackendT>
size_t DNNBackend<VendorBackendT>::get_num_internal_comms() const noexcept
{
    return m_comms.get_num_internal_comms();
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_internal_comm(size_t idx) const
    -> std::shared_ptr<InternalCommType>
{
    return m_comms.get_internal_comm(idx);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_segmented_ar_comm(size_t idx)
    -> AlCommType*
{
    return m_comms.get_segmented_ar_comm(idx);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_chanfilt_channel_comm(size_t idx)
    -> AlCommType*
{
    return m_comms.get_chanfilt_channel_comm(idx);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_chanfilt_filter_comm(size_t idx)
    -> AlCommType*
{
    return m_comms.get_chanfilt_filter_comm(idx);
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::init_chanfilt_channel_comm(size_t seg,
                                                            MPI_Comm comm)
{
    return m_comms.init_chanfilt_channel_comm(seg, comm);
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::init_chanfilt_filter_comm(size_t seg,
                                                           MPI_Comm comm)
{
    return m_comms.init_chanfilt_filter_comm(seg, comm);
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::init_segmented_ar_comm(size_t seg,
                                                        MPI_Comm comm)
{
    return m_comms.init_segmented_ar_comm(seg, comm);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_handle() const noexcept -> Handle_t
{
    return m_handle;
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_stream() const noexcept -> Stream_t
{
    return m_stream_mgr.stream();
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_internal_priority_stream(size_t idx) const
    -> Stream_t
{
    return m_stream_mgr.priority_stream(idx);
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::wait() const
{
    h2::gpu::sync(this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::activation_forward(
    ActivationDescriptor_t const& act_desc,
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* x,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* y,
    Stream_t stream) const
{
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(xdesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto input_proxy = read_proxy(handle, xdesc, x);
    auto output_proxy = write_proxy(handle, ydesc, y, beta);
    GPUDNNBackend::activation_forward(handle,
                                      act_desc,
                                      a.get(),
                                      input_proxy.desc(),
                                      input_proxy.ptr(),
                                      b.get(),
                                      output_proxy.desc(),
                                      output_proxy.ptr());

    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::activation_forward(
    ActivationDescriptor_t const& act_desc,
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* x,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* y) const
{
    return this->activation_forward(
        act_desc, alpha, xdesc, x, beta, ydesc, y, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::activation_backward(
    ActivationDescriptor_t const& act_desc,
    double alpha,
    TensorDescriptor_t const& ydesc,
    void const* y,
    TensorDescriptor_t const& dydesc,
    void const* dy,
    TensorDescriptor_t const& xdesc,
    void const* x,
    double beta,
    TensorDescriptor_t const& dxdesc,
    void* dx,
    Stream_t stream) const
{
    // The _actual_ requirement here is that "strides(m_output_d)
    // == strides(m_d_output_d) && strides(m_input_d) ==
    // strides(m_d_input_d)" We can't handle that directly, so
    // instead we proxy everything, leaving plenty of room for
    // future optimization.
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(ydesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto y_proxy = force_read_proxy(handle, ydesc, y);
    auto dy_proxy = force_read_proxy(handle, dydesc, dy);
    auto x_proxy = force_read_proxy(handle, xdesc, x);
    auto dx_proxy = force_write_proxy(handle, dxdesc, dx, beta);
    GPUDNNBackend::activation_backward(handle,
                                       act_desc,
                                       a.get(),
                                       y_proxy.desc(),
                                       y_proxy.ptr(),
                                       dy_proxy.desc(),
                                       dy_proxy.ptr(),
                                       x_proxy.desc(),
                                       x_proxy.ptr(),
                                       b.get(),
                                       dx_proxy.desc(),
                                       dx_proxy.ptr());

    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::activation_backward(
    ActivationDescriptor_t const& act_desc,
    double alpha,
    TensorDescriptor_t const& ydesc,
    void const* y,
    TensorDescriptor_t const& dydesc,
    void const* dy,
    TensorDescriptor_t const& xdesc,
    void const* x,
    double beta,
    TensorDescriptor_t const& dxdesc,
    void* dx) const
{
    return this->activation_backward(act_desc,
                                     alpha,
                                     ydesc,
                                     y,
                                     dydesc,
                                     dy,
                                     xdesc,
                                     x,
                                     beta,
                                     dxdesc,
                                     dx,
                                     this->get_stream());
}

template <typename VendorBackendT>
size_t DNNBackend<VendorBackendT>::get_conv_forward_workspace_size(
    TensorDescriptor_t const& in_desc,
    FilterDescriptor_t const& filter_desc,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& out_desc,
    ConvFwdAlgo_t const& algo) const
{
    return GPUDNNBackend::get_conv_forward_workspace_size(
        this->get_handle(),
        read_proxy(in_desc).desc(),
        filter_desc,
        conv_desc,
        write_proxy(out_desc).desc(),
        algo);
}

template <typename VendorBackendT>
size_t DNNBackend<VendorBackendT>::get_conv_bwd_data_workspace_size(
    FilterDescriptor_t const& filter_desc,
    TensorDescriptor_t const& dy_desc,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& dx_desc,
    ConvBwdDataAlgo_t const& algo) const
{
    return GPUDNNBackend::get_conv_bwd_data_workspace_size(
        this->get_handle(),
        filter_desc,
        read_proxy(dy_desc).desc(),
        conv_desc,
        write_proxy(dx_desc).desc(),
        algo);
}

template <typename VendorBackendT>
size_t DNNBackend<VendorBackendT>::get_conv_bwd_filter_workspace_size(
    TensorDescriptor_t const& in_desc,
    TensorDescriptor_t const& dy_desc,
    ConvolutionDescriptor_t const& conv_desc,
    FilterDescriptor_t const& dw_desc,
    ConvBwdFilterAlgo_t const& algo) const
{
    return GPUDNNBackend::get_conv_bwd_filter_workspace_size(
        this->get_handle(),
        read_proxy(in_desc).desc(),
        read_proxy(dy_desc).desc(),
        conv_desc,
        dw_desc,
        algo);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_fwd_algorithm(
    std::string const& name_,
    TensorDescriptor_t const& input_desc,
    void const* input,
    FilterDescriptor_t const& filter_desc,
    void const* filter,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& output_desc,
    void* output,
    size_t ws_size) const -> ConvFwdAlgo_t
{
    std::string const name = (name_ == "DEFAULT" ? "HEURISTIC" : name_);
    if (name == "HEURISTIC")
    {
        return GPUDNNBackend::get_fwd_algorithm_by_heuristics(
            this->get_handle(),
            input_desc,
            filter_desc,
            conv_desc,
            output_desc,
            ws_size);
    }
    else if (name == "AUTOTUNE")
    {
        return GPUDNNBackend::get_fwd_algorithm_by_autotune(this->get_handle(),
                                                            input_desc,
                                                            input,
                                                            filter_desc,
                                                            filter,
                                                            conv_desc,
                                                            output_desc,
                                                            output,
                                                            ws_size);
    }
    // Handles "DETERMINISTIC"
    return GPUDNNBackend::get_fwd_algorithm_by_name(name);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_bwd_data_algorithm(
    std::string const& name_,
    FilterDescriptor_t const& filter_desc,
    void const* filter,
    TensorDescriptor_t const& d_output_desc,
    void const* d_output,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& d_input_desc,
    void* d_input,
    size_t ws_size) const -> ConvBwdDataAlgo_t
{
    std::string const name = (name_ == "DEFAULT" ? "HEURISTIC" : name_);
    if (name == "HEURISTIC")
    {
        return GPUDNNBackend::get_bwd_data_algorithm_by_heuristics(
            this->get_handle(),

            filter_desc,
            d_output_desc,
            conv_desc,
            d_input_desc,
            ws_size);
    }
    else if (name == "AUTOTUNE")
    {
        return GPUDNNBackend::get_bwd_data_algorithm_by_autotune(
            this->get_handle(),
            filter_desc,
            filter,
            d_output_desc,
            d_output,
            conv_desc,
            d_input_desc,
            d_input,
            ws_size);
    }
    return GPUDNNBackend::get_bwd_data_algorithm_by_name(name);
}

template <typename VendorBackendT>
auto DNNBackend<VendorBackendT>::get_bwd_filter_algorithm(
    std::string const& name_,
    TensorDescriptor_t const& input_desc,
    void const* input,
    TensorDescriptor_t const& d_output_desc,
    void const* d_output,
    ConvolutionDescriptor_t const& conv_desc,
    FilterDescriptor_t const& d_filter_desc,
    void* d_filter,
    size_t ws_size) const -> ConvBwdFilterAlgo_t
{
    std::string const name = (name_ == "DEFAULT" ? "HEURISTIC" : name_);
    if (name == "HEURISTIC")
    {
        return GPUDNNBackend::get_bwd_filter_algorithm_by_heuristics(
            this->get_handle(),
            input_desc,
            d_output_desc,
            conv_desc,
            d_filter_desc,
            ws_size);
    }
    else if (name == "AUTOTUNE")
    {
        return GPUDNNBackend::get_bwd_filter_algorithm_by_autotune(
            this->get_handle(),
            input_desc,
            input,
            d_output_desc,
            d_output,
            conv_desc,
            d_filter_desc,
            d_filter,
            ws_size);
    }
    return GPUDNNBackend::get_bwd_filter_algorithm_by_name(name);
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_forward(
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* x,
    FilterDescriptor_t const& filter_desc,
    void const* filter_data,
    ConvolutionDescriptor_t const& conv_desc,
    ConvFwdAlgo_t const& conv_algo,
    void* workspace,
    size_t workspace_bytes,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* y,
    Stream_t stream) const
{
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(xdesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto input_proxy = read_proxy(handle, xdesc, x);
    auto output_proxy = write_proxy(handle, ydesc, y, beta);
    GPUDNNBackend::convolution_forward(handle,
                                       a.get(),
                                       input_proxy.desc(),
                                       input_proxy.ptr(),
                                       filter_desc,
                                       filter_data,
                                       conv_desc,
                                       conv_algo,
                                       workspace,
                                       workspace_bytes,
                                       b.get(),
                                       output_proxy.desc(),
                                       output_proxy.ptr());

    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_forward(
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    FilterDescriptor_t const& filter_desc,
    void const* const filter_data,
    ConvolutionDescriptor_t const& conv_desc,
    ConvFwdAlgo_t const& conv_algo,
    void* const workspace,
    size_t workspace_bytes,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* const y) const
{
    return this->convolution_forward(alpha,
                                     xdesc,
                                     x,
                                     filter_desc,
                                     filter_data,
                                     conv_desc,
                                     conv_algo,
                                     workspace,
                                     workspace_bytes,
                                     beta,
                                     ydesc,
                                     y,
                                     this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_bwd_data(
    double const alpha,
    FilterDescriptor_t const& filter_desc,
    void const* const filter_data,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    ConvolutionDescriptor_t const& conv_desc,
    ConvBwdDataAlgo_t const& conv_algo,
    void* const workspace,
    size_t const workspace_bytes,
    double const beta,
    TensorDescriptor_t const& dxdesc,
    void* const dx,
    Stream_t stream) const
{
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(dydesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto dy_proxy = read_proxy(handle, dydesc, dy);
    auto dx_proxy = write_proxy(handle, dxdesc, dx, beta);
    GPUDNNBackend::convolution_bwd_data(handle,
                                        a.get(),
                                        filter_desc,
                                        filter_data,
                                        dy_proxy.desc(),
                                        dy_proxy.ptr(),
                                        conv_desc,
                                        conv_algo,
                                        workspace,
                                        workspace_bytes,
                                        b.get(),
                                        dx_proxy.desc(),
                                        dx_proxy.ptr());
    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_bwd_data(
    double const alpha,
    FilterDescriptor_t const& filter_desc,
    void const* const filter_data,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    ConvolutionDescriptor_t const& conv_desc,
    ConvBwdDataAlgo_t const& conv_algo,
    void* const workspace,
    size_t const workspace_bytes,
    double const beta,
    TensorDescriptor_t const& dxdesc,
    void* const dx) const
{
    return this->convolution_bwd_data(alpha,
                                      filter_desc,
                                      filter_data,
                                      dydesc,
                                      dy,
                                      conv_desc,
                                      conv_algo,
                                      workspace,
                                      workspace_bytes,
                                      beta,
                                      dxdesc,
                                      dx,
                                      this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_bwd_filter(
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    ConvolutionDescriptor_t const& conv_desc,
    ConvBwdFilterAlgo_t const& conv_algo,
    void* const workspace,
    size_t const workspace_bytes,
    double const beta,
    FilterDescriptor_t const& dwdesc,
    void* const dw,
    Stream_t const stream) const
{
    // Even in cuDNN-land, filters must always be fully-packed. Hence,
    // we do not need a proxy for them.
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(xdesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto const x_proxy = read_proxy(handle, xdesc, x);
    auto const dy_proxy = read_proxy(handle, dydesc, dy);
    GPUDNNBackend::convolution_bwd_filter(handle,
                                          a.get(),
                                          x_proxy.desc(),
                                          x_proxy.ptr(),
                                          dy_proxy.desc(),
                                          dy_proxy.ptr(),
                                          conv_desc,
                                          conv_algo,
                                          workspace,
                                          workspace_bytes,
                                          b.get(),
                                          dwdesc,
                                          dw);
    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::convolution_bwd_filter(
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    ConvolutionDescriptor_t const& conv_desc,
    ConvBwdFilterAlgo_t const& conv_algo,
    void* const workspace,
    size_t const workspace_bytes,
    double const beta,
    FilterDescriptor_t const& dwdesc,
    void* const dw) const
{
    return this->convolution_bwd_filter(alpha,
                                        xdesc,
                                        x,
                                        dydesc,
                                        dy,
                                        conv_desc,
                                        conv_algo,
                                        workspace,
                                        workspace_bytes,
                                        beta,
                                        dwdesc,
                                        dw,
                                        this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::apply_fwd_bias(
    double alpha,
    TensorDescriptor_t const& bias_desc,
    void const* const bias,
    double beta,
    TensorDescriptor_t const& y_desc,
    void* y)
{
    auto const handle = this->get_handle();

    auto const dt = GPUDNNBackend::get_tensor_datatype(bias_desc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto const bias_proxy = read_proxy(handle, bias_desc, bias);
    auto const y_proxy = write_proxy(handle, y_desc, y, beta);

    VendorBackendT::apply_fwd_bias(handle,
                                   a.get(),
                                   bias_proxy.desc(),
                                   bias_proxy.ptr(),
                                   b.get(),
                                   y_proxy.desc(),
                                   y_proxy.ptr());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::apply_bwd_bias(
    double alpha,
    TensorDescriptor_t const& dy_desc,
    void const* dy_data,
    double beta,
    TensorDescriptor_t const& db_desc,
    void* db_data)
{
    auto const handle = this->get_handle();

    auto const dt = GPUDNNBackend::get_tensor_datatype(dy_desc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto const dy_proxy = read_proxy(handle, dy_desc, dy_data);
    auto const db_proxy = write_proxy(handle, db_desc, db_data, beta);

    VendorBackendT::apply_bwd_bias(handle,
                                   a.get(),
                                   dy_proxy.desc(),
                                   dy_proxy.ptr(),
                                   b.get(),
                                   db_proxy.desc(),
                                   db_proxy.ptr());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::pooling_forward(
    PoolingDescriptor_t const& pooling_desc,
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* const y,
    bool const training,
    Stream_t const stream) const
{
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(xdesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto input_prox = read_proxy(handle, xdesc, x);
    auto output_prox = write_proxy(handle, ydesc, y, beta);

    VendorBackend::pooling_forward(handle,
                                   pooling_desc,
                                   a.get(),
                                   input_prox.desc(),
                                   input_prox.ptr(),
                                   b.get(),
                                   output_prox.desc(),
                                   output_prox.ptr(),
                                   training);
    // NOTE: Workspace is managed at the VendorBackend level
    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::pooling_forward(
    PoolingDescriptor_t const& pooling_desc,
    double const alpha,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    double const beta,
    TensorDescriptor_t const& ydesc,
    void* const y,
    bool const training) const
{
    return this->pooling_forward(pooling_desc,
                                 alpha,
                                 xdesc,
                                 x,
                                 beta,
                                 ydesc,
                                 y,
                                 training,
                                 this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::pooling_backward(
    PoolingDescriptor_t const& pooling_desc,
    double const alpha,
    TensorDescriptor_t const& ydesc,
    void const* const y,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    double const beta,
    TensorDescriptor_t const& dxdesc,
    void* const dx,
    Stream_t const stream) const
{
    auto const handle = this->get_handle();
    GPUDNNBackend::set_stream(handle, stream);

    auto const dt = GPUDNNBackend::get_tensor_datatype(ydesc);
    auto const a = make_host_scalar(dt, alpha);
    auto const b = make_host_scalar(dt, beta);

    auto const output_proxy = read_proxy(handle, ydesc, y);
    auto const d_output_proxy = read_proxy(handle, dydesc, dy);
    auto const input_proxy = read_proxy(handle, xdesc, x);
    auto const d_input_proxy = write_proxy(handle, dxdesc, dx, beta);
    GPUDNNBackend::pooling_backward(handle,
                                    pooling_desc,
                                    a.get(),
                                    output_proxy.desc(),
                                    output_proxy.ptr(),
                                    d_output_proxy.desc(),
                                    d_output_proxy.ptr(),
                                    input_proxy.desc(),
                                    input_proxy.ptr(),
                                    b.get(),
                                    d_input_proxy.desc(),
                                    d_input_proxy.ptr());

    GPUDNNBackend::set_stream(handle, this->get_stream());
}

template <typename VendorBackendT>
void DNNBackend<VendorBackendT>::pooling_backward(
    PoolingDescriptor_t const& pooling_desc,
    double const alpha,
    TensorDescriptor_t const& ydesc,
    void const* const y,
    TensorDescriptor_t const& dydesc,
    void const* const dy,
    TensorDescriptor_t const& xdesc,
    void const* const x,
    double const beta,
    TensorDescriptor_t const& dxdesc,
    void* const dx) const
{
    return this->pooling_backward(pooling_desc,
                                  alpha,
                                  ydesc,
                                  y,
                                  dydesc,
                                  dy,
                                  xdesc,
                                  x,
                                  beta,
                                  dxdesc,
                                  dx,
                                  this->get_stream());
}

template class DNNBackend<GPUDNNBackend>;

} // namespace distconv

// Miscellaneous free functions

#if H2_HAS_ROCM

// These are used in (unported) benchmarks:
std::string distconv::util::get_miopen_version_number_string()
{
    size_t major, minor, patch;
    DISTCONV_CHECK_MIOPEN(miopenGetVersion(&major, &minor, &patch));
    std::ostringstream oss;
    oss << "MIOpen v" << major << "." << minor << "." << patch;
    return oss.str();
}

// "NCHW" format.
std::vector<int>
distconv::util::get_miopen_dims(int const num_samples,
                                int const num_channels,
                                std::vector<int> const& spatial_dims)
{
    std::vector<int> dims;
    dims.reserve(spatial_dims.size() + 2);
    dims.push_back(num_samples);
    dims.push_back(num_channels);
    dims.insert(dims.end(), spatial_dims.cbegin(), spatial_dims.cend());
    return dims;
}

// Only supports NCHW format; assumes fully-packed tensor.
std::vector<int>
distconv::util::get_miopen_strides(int const num_samples,
                                   int const num_channels,
                                   std::vector<int> const& spatial_dims,
                                   std::string const& fmt)
{
    if ((fmt != "NCHW") && (fmt != "NCDHW"))
        throw std::runtime_error("bad format; only NCHW,NCDHW supported.");

    size_t const num_spatial_dims = spatial_dims.size();
    assert_always(num_spatial_dims == 2 || num_spatial_dims == 3);

    return get_fully_packed_strides(
        get_miopen_dims(num_samples, num_channels, spatial_dims));
}

std::string distconv::util::get_name(miopenConvFwdAlgorithm_t const& algo)
{
    return MIOpenConvolutionFwdAlgorithms::get_name(algo);
}

std::string distconv::util::get_name(miopenConvBwdDataAlgorithm_t const& algo)
{
    return MIOpenConvolutionBwdDataAlgorithms::get_name(algo);
}

std::string
distconv::util::get_name(miopenConvBwdWeightsAlgorithm_t const& algo)
{
    return MIOpenConvolutionBwdWeightsAlgorithms::get_name(algo);
}

std::ostream& distconv::util::operator<<(std::ostream& os, miopenDataType_t& dt)
{
    switch (dt)
    {
    case miopenHalf: return os << "half";         // Fully supported
    case miopenFloat: return os << "float";       // Fully supported
    case miopenInt32: return os << "int32";       // Partially supported
    case miopenInt8: return os << "int8";         // Partially supported
    case miopenInt8x4: return os << "int8x4";     // Partially supported
    case miopenBFloat16: return os << "bfloat16"; // Partially supported
    case miopenDouble: return os << "double";     // Partially supported
    default: return os << "UNKNOWN";
    }
}

namespace
{
// I don't like the versions in distconv/util/util.hpp. I want to
// stream the output, not create a stream, copy to a string, and then
// stream the string again. What a headache.
template <typename IterT>
auto strjoin(std::ostream& os,
             IterT begin,
             IterT const end,
             char const delim = ',') -> IterT
{
    if (begin != end)
    {
        os << *begin;
        while (++begin != end)
            os << delim << *begin;
    }
    return end;
}
template <typename ContainerT>
void strjoin(std::ostream& os, ContainerT const& c, char const delim = ',')
{
    if (std::size(c) == 0UL)
        return;
    strjoin(os, std::cbegin(c), std::cend(c), delim);
}
} // namespace

std::string distconv::util::tostring(miopenTensorDescriptor_t const& desc)
{
    int num_dims = -1;
    // This API is TERRIBLY named. This actually gets the number of
    // dimensions in the tensor.
    DISTCONV_CHECK_MIOPEN(miopenGetTensorDescriptorSize(desc, &num_dims));

    miopenDataType_t dt;
    std::vector<int> dims, strides;
    dims.resize(num_dims);
    strides.resize(num_dims);

    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()));
    std::ostringstream oss;
    oss << "Tensor descriptor: #dims=" << num_dims << ", type=" << dt
        << ", dims=";
    strjoin(oss, dims, 'x');
    oss << ", strides=";
    strjoin(oss, strides, 'x');
    return oss.str();
}

std::string distconv::util::tostring(miopenConvolutionDescriptor_t const& desc)
{
    int const max_spatial_dims = 5; // MIOpen supports up to 5-D tensors.
    std::vector<int> padding, strides, dilations;
    miopenConvolutionMode_t mode;
    padding.reserve(max_spatial_dims);
    strides.reserve(max_spatial_dims);
    dilations.reserve(max_spatial_dims);

    int spatial_dims = -1;
    // This gets the correct value for spatial_dims.
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(desc,
                                                           0,
                                                           &spatial_dims,
                                                           padding.data(),
                                                           strides.data(),
                                                           dilations.data(),
                                                           &mode));
    assert_always(spatial_dims > 0 && spatial_dims <= 5);

    padding.resize(spatial_dims);
    strides.resize(spatial_dims);
    dilations.resize(spatial_dims);
    // This fills in the rest of the values.
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(desc,
                                                           spatial_dims,
                                                           &spatial_dims,
                                                           padding.data(),
                                                           strides.data(),
                                                           dilations.data(),
                                                           &mode));

    std::ostringstream oss;
    oss << "Convolution descriptor: spatial_dims=" << spatial_dims
        << ", padding=";
    strjoin(oss, padding, 'x');
    oss << ", strides=";
    strjoin(oss, strides, 'x');
    oss << ", dilations=";
    strjoin(oss, dilations, 'x');
    return oss.str();
}

#endif // H2_HAS_ROCM
