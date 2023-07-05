////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/dace_backend.hpp"

#include "distconv/util/util_mpi.hpp" // For printouts

#include <dlfcn.h>

#include <sstream>
#include <string>

namespace distconv
{
///////////////////////////////////////////////////////////////////////////
// Descriptor functionality

// DaCe-generated internal function types
typedef dacehandle_t (*initfunc_t)();

template <class C, class T>
std::basic_ostream<C, T>&
write_s5d(std::basic_ostream<C, T>& os, const s5d& s, bool ignore_first = false)
{
    if (ignore_first)
    {
        return os << "B_" << std::get<1>(s) << '_' << std::get<2>(s) << '_'
                  << std::get<3>(s) << '_' << std::get<4>(s);
    }

    return os << std::get<0>(s) << '_' << std::get<1>(s) << '_'
              << std::get<2>(s) << '_' << std::get<3>(s) << '_'
              << std::get<4>(s);
}

template <class C, class T>
std::basic_ostream<C, T>& write_convparams(std::basic_ostream<C, T>& os,
                                           const ConvParams& p)
{
    return os << p.pads[0] << '_' << p.pads[1] << '_' << p.pads[2] << '_'
              << p.strides[0] << '_' << p.strides[1] << '_' << p.strides[2]
              << '_' << p.dilation[0] << '_' << p.dilation[1] << '_'
              << p.dilation[2] << '_' << p.groups;
}

std::string ConvDescriptor::hash(bool dynamic_minibatch_size) const
{
    std::stringstream stream;
    const char* ctype =
        (this->type == FORWARD
             ? "fwd"
             : (this->type == BACKWARD_FILTER ? "bwdfilt" : "bwddata"));
    stream << "conv" << this->get_dimensionality() << "d_";
    write_s5d(stream, this->x_shape, dynamic_minibatch_size);
    stream << '_';
    write_s5d(stream, this->x_strides);
    stream << '_';
    write_s5d(stream, this->w_shape);
    stream << '_';
    write_s5d(stream, this->y_shape, dynamic_minibatch_size);
    stream << '_';
    write_s5d(stream, this->y_strides);
    stream << '_';
    write_convparams(stream, this->params);
    stream << '_' << ctype;
    return stream.str();
}

bool ConvParams::operator<(const ConvParams& other) const
{
    for (int i = 0; i < 3; ++i)
    {
        if (pads[i] < other.pads[i])
            return true;
        if (pads[i] > other.pads[i])
            return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (strides[i] < other.strides[i])
            return true;
        if (strides[i] > other.strides[i])
            return false;
    }
    for (int i = 0; i < 3; ++i)
    {
        if (dilation[i] < other.dilation[i])
            return true;
        if (dilation[i] > other.dilation[i])
            return false;
    }
    return groups < other.groups;
}

bool ConvParams::operator==(const ConvParams& other) const
{
    for (int i = 0; i < 3; ++i)
        if (pads[i] != other.pads[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (strides[i] != other.strides[i])
            return false;
    for (int i = 0; i < 3; ++i)
        if (dilation[i] != other.dilation[i])
            return false;
    return groups == other.groups;
}

bool ConvDescriptor::operator<(const ConvDescriptor& other) const
{
    if (type < other.type)
        return true;
    if (type == other.type)
    {
        if (params < other.params)
            return true;
        if (params == other.params)
        {
            if (x_shape < other.x_shape)
                return true;
            if (x_shape == other.x_shape)
            {
                if (x_strides < other.x_strides)
                    return true;
                if (x_strides == other.x_strides)
                {
                    if (w_shape < other.w_shape)
                        return true;
                    if (w_shape == other.w_shape)
                    {
                        if (y_shape < other.y_shape)
                            return true;
                        if (y_shape == other.y_shape)
                            return y_strides < other.y_strides;
                    }
                }
            }
        }
    }
    return false;
}

///////////////////////////////////////////////////////////////////////////

template <typename VendorBackendT>
DaCeDNNBackend<VendorBackendT>::DaCeDNNBackend(MPI_Comm comm,
                                               Handle_t handle,
                                               Options opts)
    : DNNBackend<VendorBackendT>(comm, handle, opts)
{}

template <typename VendorBackendT>
DaCeDNNBackend<VendorBackendT>::DaCeDNNBackend(MPI_Comm comm,
                                               Handle_t handle,
                                               Stream_t stream,
                                               Options opts)
    : DNNBackend<VendorBackendT>(comm, handle, stream, opts)
{}

template <typename VendorBackendT>
DaCeDNNBackend<VendorBackendT>::~DaCeDNNBackend()
{
    // Loop over libraries and unload them
    for (const auto& iter : m_dace_libraries)
    {
        if (!unload(iter.second))
            util::MPIPrintStreamWarning()
                << "Unable to unload library: " << iter.first.hash();
    }
}

// API

template <typename VendorBackendT>
void DaCeDNNBackend<VendorBackendT>::convolution_forward(
    double alpha,
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
    Stream_t s) const
{
    ConvDescriptor desc;
    desc.type = FORWARD;
    if (descriptor_from_tensors(xdesc, filter_desc, conv_desc, ydesc, desc))
    {
        // Try to invoke a JIT-compiled convolution
        bool jit_compiled =
            invoke(desc, x, filter_data, y, alpha, beta, workspace, s);
        if (jit_compiled)
            return;
    }

    // Fallback to vendor backend
    DNNBackend<VendorBackendT>::convolution_forward(alpha,
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
                                                    s);
}

template <typename VendorBackendT>
void DaCeDNNBackend<VendorBackendT>::convolution_bwd_data(
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
    ConvDescriptor desc;
    desc.type = BACKWARD_DATA;
    if (descriptor_from_tensors(dxdesc, filter_desc, conv_desc, dydesc, desc))
    {
        // Try to invoke a JIT-compiled convolution
        bool jit_compiled =
            invoke(desc, dx, filter_data, dy, alpha, beta, workspace, stream);
        if (jit_compiled)
            return;
    }

    // Fallback to vendor backend
    DNNBackend<VendorBackendT>::convolution_bwd_data(alpha,
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
                                                     stream);
}

template <typename VendorBackendT>
void DaCeDNNBackend<VendorBackendT>::convolution_bwd_filter(
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
    ConvDescriptor desc;
    desc.type = BACKWARD_FILTER;
    if (descriptor_from_tensors(xdesc, dwdesc, conv_desc, dydesc, desc))
    {
        // Try to invoke a JIT-compiled convolution
        bool jit_compiled =
            invoke(desc, x, dw, dy, alpha, beta, workspace, stream);
        if (jit_compiled)
            return;
    }

    // Fallback to vendor backend
    DNNBackend<VendorBackendT>::convolution_bwd_filter(alpha,
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
                                                       stream);
}

// Workspace methods

template <typename VendorBackendT>
size_t DaCeDNNBackend<VendorBackendT>::get_conv_forward_workspace_size(
    TensorDescriptor_t const& in_desc,
    FilterDescriptor_t const& filter_desc,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& out_desc,
    ConvFwdAlgo_t const& algo) const
{
    ConvDescriptor desc;
    desc.type = FORWARD;
    if (descriptor_from_tensors(
            in_desc, filter_desc, conv_desc, out_desc, desc))
    {
        // Try to get workspace size from JIT-compiled library
        dace_state library;
        bool jit_compiled = load_library_or_fallback(desc, library);
        if (jit_compiled) // Library found
        {
            if (library.get_ws_size)
                return library.get_ws_size(library.handle);
            if (library.dynbatch_get_ws_size)
                return library.dynbatch_get_ws_size(library.handle,
                                                    std::get<0>(desc.x_shape));
            // No workspace requirements
            return 0;
        }
    }

    // Any other case - fallback to vendor backend
    return DNNBackend<VendorBackendT>::get_conv_forward_workspace_size(
        in_desc, filter_desc, conv_desc, out_desc, algo);
}

template <typename VendorBackendT>
size_t DaCeDNNBackend<VendorBackendT>::get_conv_bwd_data_workspace_size(
    FilterDescriptor_t const& filter_desc,
    TensorDescriptor_t const& dy_desc,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& dx_desc,
    ConvBwdDataAlgo_t const& algo) const
{
    ConvDescriptor desc;
    desc.type = BACKWARD_DATA;
    if (descriptor_from_tensors(dx_desc, filter_desc, conv_desc, dy_desc, desc))
    {
        // Try to get workspace size from JIT-compiled library
        dace_state library;
        bool jit_compiled = load_library_or_fallback(desc, library);
        if (jit_compiled) // Library found
        {
            if (library.get_ws_size)
                return library.get_ws_size(library.handle);
            if (library.dynbatch_get_ws_size)
                return library.dynbatch_get_ws_size(library.handle,
                                                    std::get<0>(desc.x_shape));
            // No workspace requirements
            return 0;
        }
    }

    // Any other case - fallback to vendor backend
    return DNNBackend<VendorBackendT>::get_conv_bwd_data_workspace_size(
        filter_desc, dy_desc, conv_desc, dx_desc, algo);
}

template <typename VendorBackendT>
size_t DaCeDNNBackend<VendorBackendT>::get_conv_bwd_filter_workspace_size(
    TensorDescriptor_t const& in_desc,
    TensorDescriptor_t const& dy_desc,
    ConvolutionDescriptor_t const& conv_desc,
    FilterDescriptor_t const& dw_desc,
    ConvBwdFilterAlgo_t const& algo) const
{
    ConvDescriptor desc;
    desc.type = BACKWARD_FILTER;
    if (descriptor_from_tensors(in_desc, dw_desc, conv_desc, dy_desc, desc))
    {
        // Try to get workspace size from JIT-compiled library
        dace_state library;
        bool jit_compiled = load_library_or_fallback(desc, library);
        if (jit_compiled) // Library found
        {
            if (library.get_ws_size)
                return library.get_ws_size(library.handle);
            if (library.dynbatch_get_ws_size)
                return library.dynbatch_get_ws_size(library.handle,
                                                    std::get<0>(desc.x_shape));
            // No workspace requirements
            return 0;
        }
    }

    // Any other case - fallback to vendor backend
    return DNNBackend<VendorBackendT>::get_conv_bwd_filter_workspace_size(
        in_desc, dy_desc, conv_desc, dw_desc, algo);
}

// Internal methods

template <typename VendorBackendT>
bool DaCeDNNBackend<VendorBackendT>::descriptor_from_tensors(
    TensorDescriptor_t const& xdesc,
    FilterDescriptor_t const& filter_desc,
    ConvolutionDescriptor_t const& conv_desc,
    TensorDescriptor_t const& ydesc,
    ConvDescriptor& result) const
{
    int ndims = VendorBackendT::get_tensor_rank(xdesc);
    if (ndims > 5)
    {
        if (this->m_opts.jit_verbose)
        {
            util::MPIPrintStreamInfo() << "Unsupported number of dimensions in "
                                       << "convolution: " << ndims;
        }
        return false;
    }
    int xshape[5] = {0}, xstrides[5] = {0}, wshape[5] = {0}, yshape[5] = {0},
        ystrides[5] = {0};
    ConvolutionMode_t unused_mode;
    DataType_t datatype;

    VendorBackendT::get_tensor_descriptor(
        xdesc, datatype, ndims, xshape, xstrides);
    VendorBackendT::get_filter_descriptor(filter_desc, datatype, ndims, wshape);
    VendorBackendT::get_tensor_descriptor(
        ydesc, datatype, ndims, yshape, ystrides);

    // Set tuples according to arrays
    result.x_shape = s5d{xshape[0], xshape[1], xshape[2], xshape[3], xshape[4]};
    result.x_strides =
        s5d{xstrides[0], xstrides[1], xstrides[2], xstrides[3], xstrides[4]};
    result.w_shape = s5d{wshape[0], wshape[1], wshape[2], wshape[3], wshape[4]};
    result.y_shape = s5d{yshape[0], yshape[1], yshape[2], yshape[3], yshape[4]};
    result.y_strides =
        s5d{ystrides[0], ystrides[1], ystrides[2], ystrides[3], ystrides[4]};

    VendorBackendT::get_convolution_descriptor(conv_desc,
                                               ndims - 2,
                                               result.params.pads,
                                               result.params.strides,
                                               result.params.dilation,
                                               result.params.groups,
                                               unused_mode,
                                               datatype);

#if H2_HAS_ROCM
    if (datatype != miopenFloat)
#elif H2_HAS_CUDA
    if (datatype != CUDNN_DATA_FLOAT)
#else
    if (false)
#endif
    {
        // TODO(later): Add support for more data types
        if (this->m_opts.jit_verbose)
        {
            util::MPIPrintStreamInfo() << "Unsupported data type in "
                                       << "convolution: " << datatype;
        }
        return false;
    }

    return true;
}

template <typename VendorBackendT>
dace_state
DaCeDNNBackend<VendorBackendT>::try_load(const std::string& hash,
                                         bool dynamic_minibatch_size) const
{
    dace_state result;
    const std::string& path = this->m_opts.jit_cache_path;
    if (path.size() == 0)
    {
        util::MPIPrintStreamError()
            << "ERROR finding build path. Please "
            << "set the DISTCONV_JIT_CACHEPATH environment variable.";
        std::abort();
    }

    std::stringstream fname, initname, exitname, funcname;
    fname << path << "/.dacecache/" << hash << "/build/lib" << hash << ".so";
    initname << "__dace_init_" << hash;
    exitname << "__dace_exit_" << hash;
    funcname << "__program_" << hash;
    void* handle = dlopen(fname.str().c_str(), RTLD_LAZY);
    if (handle)
    {
        initfunc_t initfunc =
            (initfunc_t) dlsym(handle, initname.str().c_str());
        result.library = handle;
        if (!initfunc) // No initializer
        {
            util::MPIPrintStreamError() << "Initializer not found.";
            std::abort();
        }

        dace_exitfunc_t exitfunc =
            (dace_exitfunc_t) dlsym(result.library, exitname.str().c_str());
        if (!exitfunc) // No destructor
        {
            util::MPIPrintStreamError() << "Destructor not found.";
            std::abort();
        }
        result.dtor = exitfunc;

        // Initialize handle
        dacehandle_t dacehandle = initfunc();
        result.handle = dacehandle;

        // Set the current stream
        dace_setstream_t setstreamfunc =
            (dace_setstream_t) dlsym(handle, "__dace_gpu_set_all_streams");
        if (!setstreamfunc)
        { // No external stream setter
            util::MPIPrintStreamError()
                << "Set stream function not found, please regenerate the code.";
            std::abort();
        }
        result.setstream_func = setstreamfunc;

        // Gets workspace size and sets external pointer
        void* getwssize =
            dlsym(handle, "__dace_get_external_memory_size_GPU_Global");
        void* setws = dlsym(handle, "__dace_set_external_memory_GPU_Global");
        if (getwssize)
        { // In case of no workspace requirements, the functions will not exist
            if (!setws)
            {
                util::MPIPrintStreamError()
                    << "Get workspace size function found, but setting the "
                    << "workspace was not.";
                std::abort();
            }
            if (dynamic_minibatch_size)
            {
                result.dynbatch_get_ws_size =
                    (dynbatch_getworkspacesize_t) getwssize;
                result.dynbatch_set_workspace = (dynbatch_setworkspace_t) setws;
            }
            else
            {
                result.get_ws_size = (getworkspacesize_t) getwssize;
                result.set_workspace = (setworkspace_t) setws;
            }
        }
        result.setstream_func = setstreamfunc;

        void* func = dlsym(handle, funcname.str().c_str());
        if (dynamic_minibatch_size)
            result.dynbatch_func = (dynbatch_daceprogram_t) func;
        else
            result.func = (daceprogram_t) func;
    }
    return result;
}

template <typename VendorBackendT>
bool DaCeDNNBackend<VendorBackendT>::unload(dace_state library)
{
    if (!library.library) // No library loaded
        return true;

    // Malformed library state
    if (!library.dtor || !library.handle)
    {
        dlclose(library.library);
        return false;
    }
    library.dtor(library.handle);
    if (dlclose(library.library))
        return false;
    return true;
}

template <typename VendorBackendT>
bool DaCeDNNBackend<VendorBackendT>::invoke(const ConvDescriptor& desc,
                                            void const* x,
                                            void const* w,
                                            void const* y,
                                            float alpha,
                                            float beta,
                                            void* workspace,
                                            Stream_t stream) const
{
    dace_state library;
    this->load_library_or_fallback(desc, library);

    // No library found - fall back to vendor implementation
    if (!library.library)
        return false;

    // Set stream
    library.setstream_func(library.handle, stream);

    // Call library
    if (!library.dynbatch_func)
    {
        // Set workspace pointer as necessary
        if (library.set_workspace)
            library.set_workspace(library.handle, workspace);

        library.func(library.handle, w, x, y, alpha, beta);
    }
    else
    {
        int B = std::get<0>(desc.x_shape);

        // Set workspace pointer as necessary
        if (library.dynbatch_set_workspace)
            library.dynbatch_set_workspace(library.handle, workspace, B);

        library.dynbatch_func(library.handle, w, x, y, B, alpha, beta);
    }

    return true;
}

template <typename VendorBackendT>
bool DaCeDNNBackend<VendorBackendT>::load_library_or_fallback(
    const ConvDescriptor& desc, dace_state& library) const
{
    auto iter = m_dace_libraries.find(desc);

    // First encounter of descriptor: need to try to find a JITted library
    if (iter == m_dace_libraries.end())
    {
        // First, try to load the library with a specialized minibatch size
        std::string hash = desc.hash();
        library = try_load(hash, false);

        // If failed, try to load the library with a dynamic minibatch size
        if (!library.library)
        {
            hash = desc.hash(true);
            library = try_load(hash, true);
        }

        if (!library.library && this->m_opts.jit_verbose)
        {
            util::MPIPrintStreamInfo()
                << "JIT-compiled convolution not found for " << hash;
        }

        m_dace_libraries[desc] = library;
    }
    else
    {
        library = iter->second;
    }

    if (!library.library)
        return false;
    return true;
}

///////////////////////////////////////////////////////////////////////////

// Instantiate class with GPU backend
template class DaCeDNNBackend<GPUDNNBackend>;

} // namespace distconv
