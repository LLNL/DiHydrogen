////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "distconv/util/util_mpi.hpp"

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
};

// 5D shape
using s5d = std::tuple<int, int, int, int, int>;

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
};

using state_and_func = std::tuple<void*, void*>;

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

    state_and_func try_load(const std::string& hash);
    bool unload(const std::string& hash, state_and_func library);

    void set_stream(backend::Handle_t handle, backend::Stream_t stream);

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
        printf("FWD\n");
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
        printf("BWD-D\n");
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
        printf("BWD-F\n");
    }

protected:
    DaCeOptions m_daceopts;

    // Data descriptor repository
    std::map<backend::TensorDescriptor_t, s5d> m_shapes;
    std::map<backend::TensorDescriptor_t, s5d> m_strides;

    // Convolution descriptor repository
    std::map<backend::ConvolutionDescriptor_t, ConvParams> m_convs;

    // JIT-compiled libraries
    std::map<ConvDescriptor, state_and_func> m_dace_libraries;

    backend::Stream_t m_curstream;
};

} // namespace distconv
