////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/dnn_backend.hpp"

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

#include "./dnn_lib_utils.hpp"

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

namespace
{
size_t getenv_num_streams(size_t default_nstreams = 8)
{
    size_t nstreams = default_nstreams;
    if (char const* env = std::getenv("H2_DISTCONV_NUM_INTERNAL_STREAMS"))
        nstreams = std::stoul(env);
    return std::max(nstreams, static_cast<size_t>(0UL));
}
} // namespace

template <typename VendorBackendT>
DNNBackend<VendorBackendT>::DNNBackend(MPI_Comm comm,
                                       Handle_t handle,
                                       Options opts)
    : m_opts{std::move(opts)},
      m_handle{handle},
      m_stream_mgr{getenv_num_streams(/*default_nstreams=*/8UL)},
      m_comms{comm, m_stream_mgr}
{}

template <typename VendorBackendT>
DNNBackend<VendorBackendT>::DNNBackend(MPI_Comm comm,
                                       Handle_t handle,
                                       Stream_t stream,
                                       Options opts)
    : m_opts{std::move(opts)},
      m_handle{handle},
      m_stream_mgr{getenv_num_streams(/*default_nstreams=*/8UL), stream},
      m_comms{comm, m_stream_mgr}
{}

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

    // Handle in-place.
    //
    // NOTE: This assumes that the "output" will be correct, as that's
    // the value one is more likely to have flowing backward.
    if (x == y)
    {
        auto y_proxy = force_read_proxy(handle, ydesc, y);
        auto dy_proxy = force_read_proxy(handle, dydesc, dy);
        auto dx_proxy = force_write_proxy(handle, dxdesc, dx, beta);
        GPUDNNBackend::activation_backward(handle,
                                           act_desc,
                                           a.get(),
                                           y_proxy.desc(),
                                           y_proxy.ptr(),
                                           dy_proxy.desc(),
                                           dy_proxy.ptr(),
                                           y_proxy.desc(),
                                           y_proxy.ptr(),
                                           b.get(),
                                           dx_proxy.desc(),
                                           dx_proxy.ptr());
    }
    else
    {
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
    }

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
