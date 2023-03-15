////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "distconv/tensor/tensor.hpp" // For the tensor namespace
#include "distconv/util/util.hpp"     // For reverse
#include "distconv/util/util_mpi.hpp" // For MPIRootPrintStream
#include "distconv/vector.hpp"        // For {Int, Index}Vector

#include <Al.hpp>

#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>

namespace distconv
{
enum ConvType
{
    FORWARD = 0,
    BACKWARD_FILTER,
    BACKWARD_DATA
};

struct ConvParams
{
    int pads[3];
    int strides[3];
    int dilation[3];
    int groups;

    bool operator<(const ConvParams& other) const;
    bool operator==(const ConvParams& other) const;
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

    std::string hash() const;

    bool operator<(const ConvDescriptor& other) const;
};

typedef void* dacehandle_t;
// Not really void const* but this allows us to use one handle type for fwd/bwd
typedef void (*daceprogram_t)(dacehandle_t handle,
                              void const* w,
                              void const* x,
                              void const* y,
                              float alpha,
                              float beta);
struct dace_state
{
    void* library;
    dacehandle_t handle;
    daceprogram_t func;
};

struct DaCeOptions : public backend::Options
{
    bool m_verbose;
    std::string m_cachepath;

    DaCeOptions(bool overlap_halo_exchange = false,
                bool deterministic = false,
                bool enable_profiling = false,
                bool ws_capacity_factor = 1.0,
                bool verbose = true,
                const char* cachepath = ".jitcache")
        : Options(overlap_halo_exchange,
                  deterministic,
                  enable_profiling,
                  ws_capacity_factor),
          m_verbose(verbose),
          m_cachepath(cachepath)
    {
        set_by_environment_variables();
    }
    void set_by_environment_variables()
    {
        if (std::getenv("DACEDCONV_VERBOSE"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DACEDCONV_VERBOSE"
                                            << " detected";
            m_verbose = true;
        }
        if (std::getenv("DACEDCONV_CACHEPATH"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DACEDCONV_CACHEPATH"
                                            << " detected";
            m_cachepath = std::getenv("DACEDCONV_CACHEPATH");
        }
    }
};

// Backend context
class BackendDaCe : public BackendDNNLib_
{
public:
    BackendDaCe(MPI_Comm comm,
                backend::Handle_t handle,
                const DaCeOptions& opts = DaCeOptions())
        : BackendDNNLib_(comm, handle, opts),
          m_daceopts(opts),
          m_curstream(nullptr)
    {}

    BackendDaCe(MPI_Comm comm,
                backend::Handle_t handle,
                backend::Stream_t stream,
                const DaCeOptions& opts = DaCeOptions())
        : BackendDNNLib_(comm, handle, stream, opts),
          m_daceopts(opts),
          m_curstream(nullptr)
    {}

    virtual ~BackendDaCe()
    {
        // Loop over libraries and unload them
        for (const auto& iter : m_dace_libraries)
        {
            std::string hash = iter.first.hash();
            if (!unload(hash, iter.second))
                util::MPIPrintStreamWarning()
                    << "Unable to unload library: " << hash;
        }
    }

    std::string get_name() const
    {
        return BackendDNNLib_::get_name() + std::string("_DaCe");
    }

    dace_state compile(const ConvDescriptor& desc, const std::string& hash);
    dace_state try_load(const std::string& hash);
    bool unload(const std::string& hash, dace_state library);
    bool invoke(const ConvDescriptor& desc,
                void const* x,
                void const* w,
                void const* y,
                float alpha,
                float beta,
                void* workspace);

    void set_stream(backend::Handle_t handle, backend::Stream_t stream);

    //////////////////////////////////////////////////////////////////////////
    // Tensor/convolution descriptor management

    template <typename Tensor>
    inline void setup_filter_descriptor(backend::FilterDescriptor_t& desc,
                                        Tensor const& tensor)
    {
        int ashape[5] = {0};

        BackendDNNLib_::setup_filter_descriptor(desc, tensor);

        const std::vector<int> shape =
            tensor.get_local_real_shape().template get_vector<int>();

        if (shape.size() > 5)
        {
            util::MPIPrintStreamError()
                << "Shape dimensionality for filter is too large: "
                << shape.size();
            return;
        }

        memcpy(ashape, shape.data(), sizeof(int) * shape.size());
        m_shapes[(backend::TensorDescriptor_t) desc] = std::make_tuple(
            ashape[0], ashape[1], ashape[2], ashape[3], ashape[4]);
    }
    inline void copy_filter_descriptor(backend::FilterDescriptor_t& dst,
                                       const backend::FilterDescriptor_t& src)
    {
        m_shapes[(backend::TensorDescriptor_t) dst] =
            m_shapes[(backend::TensorDescriptor_t) src];
        m_strides[(backend::TensorDescriptor_t) dst] =
            m_strides[(backend::TensorDescriptor_t) src];
        BackendDNNLib_::copy_filter_descriptor(dst, src);
    }

    virtual void
    setup_tensor_descriptor_internal(backend::TensorDescriptor_t& desc,
                                     backend::DataType_t dt,
                                     const std::vector<int>& shape,
                                     const std::vector<int>& strides) override
    {
        int ashape[5] = {0};
        int astrides[5] = {0};
        BackendDNNLib_::setup_tensor_descriptor_internal(
            desc, dt, shape, strides);

        size_t ndims = shape.size();
        if (ndims > 5)
        {
            util::MPIPrintStreamError()
                << "Shape dimensionality for tensor is too large: " << ndims;
            return;
        }
        memcpy(ashape, shape.data(), sizeof(int) * ndims);
        memcpy(astrides, strides.data(), sizeof(int) * ndims);

        m_shapes[desc] = std::make_tuple(
            ashape[0], ashape[1], ashape[2], ashape[3], ashape[4]);
        m_strides[desc] = std::make_tuple(
            astrides[0], astrides[1], astrides[2], astrides[3], astrides[4]);
    }

    inline void
    set_tensor_dimension(backend::TensorDescriptor_t& desc, int d, int n)
    {
        // Strides stay the same, shape dimension changes
        set_value(m_shapes[desc], d, n);

        BackendDNNLib_::set_tensor_dimension(desc, d, n);
    }

    inline void copy_tensor_descriptor(backend::TensorDescriptor_t& dst,
                                       const backend::TensorDescriptor_t& src)
    {
        m_shapes[dst] = m_shapes[src];
        m_strides[dst] = m_strides[src];
        BackendDNNLib_::copy_tensor_descriptor(dst, src);
    }

    inline void
    set_convolution_descriptor(backend::ConvolutionDescriptor_t& conv_desc,
                               int const array_len,
                               int const* const pad,
                               int const* const stride,
                               int const* const dilation,
                               backend::ConvolutionMode_t const& mode,
                               backend::DataType_t const& data_type)
    {
        BackendDNNLib_::set_convolution_descriptor(
            conv_desc, array_len, pad, stride, dilation, mode, data_type);

        // Store convolution parameters
        if (array_len <= 0 || array_len > 3)
        {
            util::MPIPrintStreamError()
                << "Convolution dimensionality is malformed: " << array_len;
            return;
        }

        ConvParams p;
        memset(&p, 0, sizeof(ConvParams));
        memcpy(p.pads, pad, array_len * sizeof(int));
        memcpy(p.strides, stride, array_len * sizeof(int));
        memcpy(p.dilation, dilation, array_len * sizeof(int));

        m_convs[conv_desc] = p;
    }

    inline void
    copy_convolution_descriptor(backend::ConvolutionDescriptor_t& dst,
                                const backend::ConvolutionDescriptor_t& src)
    {
        m_convs[dst] = m_convs[src];
        BackendDNNLib_::copy_convolution_descriptor(dst, src);
    }

    inline void
    set_convolution_group_count(backend::ConvolutionDescriptor_t const& desc,
                                int ngrps)
    {
        m_convs[desc].groups = ngrps;
        BackendDNNLib_::set_convolution_group_count(desc, ngrps);
    }

    //////////////////////////////////////////////////////////////////////////
    // Convolution invocation

    template <typename T>
    void convolution_forward(backend::Handle_t handle,
                             T const& alpha,
                             backend::TensorDescriptor_t const& in_desc,
                             void const* in_data,
                             backend::FilterDescriptor_t const& filter_desc,
                             void const* filter_data,
                             backend::ConvolutionDescriptor_t const& conv_desc,
                             backend::ConvFwdAlgo_t const& conv_algo,
                             void* work_data,
                             size_t work_data_size,
                             T const& beta,
                             backend::TensorDescriptor_t const& out_desc,
                             void* out_data)
    {
        ConvDescriptor desc;
        desc.x_shape = m_shapes[in_desc];
        desc.x_strides = m_strides[in_desc];
        desc.w_shape = m_shapes[(backend::TensorDescriptor_t) filter_desc];
        desc.y_shape = m_shapes[out_desc];
        desc.y_strides = m_strides[out_desc];
        desc.type = FORWARD;
        desc.params = m_convs[conv_desc];
        if (!invoke(desc,
                    in_data,
                    filter_data,
                    out_data,
                    float(alpha),
                    float(beta),
                    work_data))
            BackendDNNLib_::convolution_forward(handle,
                                                alpha,
                                                in_desc,
                                                in_data,
                                                filter_desc,
                                                filter_data,
                                                conv_desc,
                                                conv_algo,
                                                work_data,
                                                work_data_size,
                                                beta,
                                                out_desc,
                                                out_data);
    }

    template <typename T>
    void convolution_bwd_data(backend::Handle_t handle,
                              T const& alpha,
                              backend::FilterDescriptor_t const& filter_desc,
                              void const* filter_data,
                              backend::TensorDescriptor_t const& dy_desc,
                              void const* dy_data,
                              backend::ConvolutionDescriptor_t const& conv_desc,
                              backend::ConvBwdDataAlgo_t const& conv_algo,
                              void* work_data,
                              size_t work_data_size,
                              T const& beta,
                              backend::TensorDescriptor_t const& dx_desc,
                              void* dx_data)
    {
        ConvDescriptor desc;
        desc.x_shape = m_shapes[dx_desc];
        desc.x_strides = m_strides[dx_desc];
        desc.w_shape = m_shapes[(backend::TensorDescriptor_t) filter_desc];
        desc.y_shape = m_shapes[dy_desc];
        desc.y_strides = m_strides[dy_desc];
        desc.type = BACKWARD_DATA;
        desc.params = m_convs[conv_desc];
        if (!invoke(desc,
                    dx_data,
                    filter_data,
                    dy_data,
                    float(alpha),
                    float(beta),
                    work_data))
            BackendDNNLib_::convolution_bwd_data(handle,
                                                 alpha,
                                                 filter_desc,
                                                 filter_data,
                                                 dy_desc,
                                                 dy_data,
                                                 conv_desc,
                                                 conv_algo,
                                                 work_data,
                                                 work_data_size,
                                                 beta,
                                                 dx_desc,
                                                 dx_data);
    }

    template <typename T>
    void
    convolution_bwd_filter(backend::Handle_t handle,
                           T const& alpha,
                           backend::TensorDescriptor_t const& in_desc,
                           void const* in_data,
                           backend::TensorDescriptor_t const& dy_desc,
                           void const* dy_data,
                           backend::ConvolutionDescriptor_t const& conv_desc,
                           backend::ConvBwdFilterAlgo_t const& conv_algo,
                           void* work_data,
                           size_t work_data_size,
                           T const& beta,
                           backend::FilterDescriptor_t const& dw_desc,
                           void* dw_data)
    {
        ConvDescriptor desc;
        desc.x_shape = m_shapes[in_desc];
        desc.x_strides = m_strides[in_desc];
        desc.w_shape = m_shapes[(backend::TensorDescriptor_t) dw_desc];
        desc.y_shape = m_shapes[dy_desc];
        desc.y_strides = m_strides[dy_desc];
        desc.type = BACKWARD_FILTER;
        desc.params = m_convs[conv_desc];
        if (!invoke(desc,
                    in_data,
                    dw_data,
                    dy_data,
                    float(alpha),
                    float(beta),
                    work_data))
            BackendDNNLib_::convolution_bwd_filter(handle,
                                                   alpha,
                                                   in_desc,
                                                   in_data,
                                                   dy_desc,
                                                   dy_data,
                                                   conv_desc,
                                                   conv_algo,
                                                   work_data,
                                                   work_data_size,
                                                   beta,
                                                   dw_desc,
                                                   dw_data);
    }

protected:
    DaCeOptions m_daceopts;

    // Data descriptor repository
    std::map<backend::TensorDescriptor_t, s5d> m_shapes;
    std::map<backend::TensorDescriptor_t, s5d> m_strides;

    // Convolution descriptor repository
    std::map<backend::ConvolutionDescriptor_t, ConvParams> m_convs;

    // JIT-compiled libraries
    std::map<ConvDescriptor, dace_state> m_dace_libraries;

    backend::Stream_t m_curstream;
};

} // namespace distconv
