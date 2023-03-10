////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#include "distconv/base.hpp"
#include "distconv/layers.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_miopen.hpp"
#include "distconv/util/util_rocm.hpp"

#ifdef DISTCONV_HAS_P2P
#include "p2p/p2p.hpp"
#endif // DISTCONV_HAS_P2P

#include <Al.hpp>

#include <memory>
#include <string>
#include <type_traits>
#include <unordered_set>

#include <miopen/miopen.h>

namespace distconv
{
namespace miopen
{

using ActivationDescriptor_t = miopenActivationDescriptor_t;
using ConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
using ConvolutionMode_t = miopenConvolutionMode_t;
using DataType_t = miopenDataType_t;
using ConvFwdAlgo_t = miopenConvFwdAlgorithm_t;
using ConvBwdDataAlgo_t = miopenConvBwdDataAlgorithm_t;
using ConvBwdFilterAlgo_t = miopenConvBwdWeightsAlgorithm_t;
using FilterDescriptor_t = miopenTensorDescriptor_t;
using PoolingDescriptor_t = miopenPoolingDescriptor_t;
using PoolingMode_t = miopenPoolingMode_t;
using TensorDescriptor_t = miopenTensorDescriptor_t;
using Handle_t = miopenHandle_t;
using Stream_t = hipStream_t;

// TODO: Move to the runtime stuff.
using Event_t = hipEvent_t;
inline Event_t make_event()
{
    return h2::gpu::make_event();
}
inline void destroy_event(hipEvent_t const& event)
{
    h2::gpu::destroy(event);
}
inline void record_event(hipEvent_t const& event, hipStream_t const& stream)
{
    DISTCONV_CHECK_HIP(hipEventRecord(event, stream));
}
inline float elapsed_time(hipEvent_t const& start, hipEvent_t const& end)
{
    float elapsed;
    DISTCONV_CHECK_HIP(hipEventElapsedTime(&elapsed, start, end));
    return elapsed;
}
inline size_t get_available_memory()
{
    return h2::gpu::mem_info().free;
}

inline Handle_t make_handle()
{
    miopenHandle_t handle;
    DISTCONV_CHECK_MIOPEN(miopenCreate(&handle));
    return handle;
}

inline void destroy_handle(miopenHandle_t handle)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroy(handle));
}

inline miopenTensorDescriptor_t make_tensor_descriptor()
{
    miopenTensorDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc));
    return desc;
}

inline void destroy_tensor_descriptor(miopenTensorDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyTensorDescriptor(desc));
}

inline miopenTensorDescriptor_t make_filter_descriptor()
{
    return make_tensor_descriptor();
}

inline void destroy_filter_descriptor(miopenTensorDescriptor_t const& desc)
{
    destroy_tensor_descriptor(desc);
}

template <typename T>
void print_array(T const* const data,
                 size_t const size,
                 std::ostream& os = std::cout)
{
    os << "[";
    for (size_t ii = 0; ii < size; ++ii)
        os << " " << data[ii];
    os << " ]";
}

template <typename Tensor>
inline void setup_filter_descriptor(FilterDescriptor_t& desc,
                                    Tensor const& tensor)
{
    auto const dt = util::get_miopen_type<typename Tensor::data_type>();
    int_vector const shape =
        tensor.get_local_real_shape().template get_vector<int>();
    std::vector<int> strides;
    strides.reserve(shape.size());
    strides.push_back(1);
    std::partial_sum(shape.begin(),
                     shape.end() - 1,
                     std::back_inserter(strides),
                     std::multiplies<int>());
    std::reverse(begin(strides), end(strides));
    DISTCONV_CHECK_MIOPEN(miopenSetTensorDescriptor(
        desc, dt, shape.size(), util::reverse(shape).data(), strides.data()));
}

template <typename Tensor, typename ShapeType>
inline void setup_tensor_descriptor(miopenTensorDescriptor_t& desc,
                                    Tensor const& tensor,
                                    ShapeType const& shape)
{
    miopenDataType_t const dt =
        util::get_miopen_type<typename Tensor::data_type>();
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

    DISTCONV_CHECK_MIOPEN(miopenSetTensorDescriptor(
        desc,
        dt,
        shape.num_dims(),
        util::reverse(IntVector(shape)).data(),
        util::reverse(strides).get_vector<int>().data()));
}

template <typename Tensor>
inline void setup_tensor_descriptor(miopenTensorDescriptor_t& desc,
                                    Tensor const& tensor,
                                    IntVector const& halo_fwd,
                                    IntVector const& halo_bwd)
{
    auto shape = tensor.get_local_shape();
    shape = shape + tensor::Shape(halo_fwd) + tensor::Shape(halo_bwd);
    return setup_tensor_descriptor(desc, tensor, shape);
}

template <typename Tensor>
inline void setup_tensor_descriptor(miopenTensorDescriptor_t& desc,
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
inline void setup_tensor_descriptor(miopenTensorDescriptor_t& desc,
                                    Tensor const& tensor,
                                    bool include_halo = true)
{
    std::vector<bool> include_halo_array(tensor.get_num_dims(), include_halo);
    setup_tensor_descriptor(
        desc, tensor, include_halo_array, include_halo_array);
}

inline int get_tensor_rank(miopenTensorDescriptor_t const& desc)
{
    int num_dims = -1;
    // This API is TERRIBLY named. This actually gets the number of
    // dimensions in the tensor.
    DISTCONV_CHECK_MIOPEN(miopenGetTensorDescriptorSize(desc, &num_dims));
    return num_dims;
}

inline int get_tensor_dimension(miopenTensorDescriptor_t const& desc, int d)
{
    int const num_dims = get_tensor_rank(desc);
    d = d < 0 ? num_dims + d : d;
    assert_always(d < num_dims);

    miopenDataType_t dt;
    std::vector<int> dims, strides;
    dims.reserve(num_dims);
    strides.reserve(num_dims);
    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()));
    return dims[num_dims - d - 1];
}

inline void set_tensor_dimension(miopenTensorDescriptor_t& desc, int d, int n)
{
    int const num_dims = get_tensor_rank(desc);
    d = d < 0 ? num_dims + d : d;
    assert_always(d < num_dims);

    miopenDataType_t dt;
    std::vector<int> dims, strides;
    dims.reserve(num_dims);
    strides.reserve(num_dims);

    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()));
    dims[num_dims - d - 1] = n;
    // FIXME (TRB): Need to recompute strides??
    DISTCONV_CHECK_MIOPEN(miopenSetTensorDescriptor(
        desc, dt, num_dims, dims.data(), strides.data()));
}

inline int get_tensor_num_dimensions(miopenTensorDescriptor_t const& desc)
{
    return get_tensor_rank(desc);
}

inline void set_tensor_num_samples(miopenTensorDescriptor_t& desc, int n)
{
    int const num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    set_tensor_dimension(desc, num_sample_dim, n);
}

inline int get_tensor_num_samples(miopenTensorDescriptor_t const& desc)
{
    int const num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    return get_tensor_dimension(desc, num_sample_dim);
}

inline void copy_tensor_descriptor(miopenTensorDescriptor_t& dst,
                                   miopenTensorDescriptor_t const& src)
{
    auto const num_dims = get_tensor_rank(src);
    miopenDataType_t dt;
    std::vector<int> dims, strides;
    dims.reserve(num_dims);
    strides.reserve(num_dims);

    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(src, &dt, dims.data(), strides.data()));

    DISTCONV_CHECK_MIOPEN(miopenSetTensorDescriptor(
        dst, dt, num_dims, dims.data(), strides.data()));
}

inline void copy_filter_descriptor(miopenTensorDescriptor_t& dst,
                                   miopenTensorDescriptor_t const& src)
{
    copy_tensor_descriptor(dst, src);
}

template <int ND>
inline int get_filter_descriptor_dimension(miopenTensorDescriptor_t const& desc,
                                           int d)
{
    return get_tensor_dimension(desc, d);
}

inline miopenConvolutionDescriptor_t make_convolution_descriptor()
{
    miopenConvolutionDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&desc));
    return desc;
}

inline void
destroy_convolution_descriptor(miopenConvolutionDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyConvolutionDescriptor(desc));
}

inline void
set_convolution_group_count(miopenConvolutionDescriptor_t const& desc,
                            int ngrps)
{
    DISTCONV_CHECK_MIOPEN(miopenSetConvolutionGroupCount(desc, ngrps));
}

inline void set_convolution_descriptor(ConvolutionDescriptor_t& conv_desc,
                                       int const array_len,
                                       int const* const pad,
                                       int const* const stride,
                                       int const* const dilation,
                                       ConvolutionMode_t const& mode,
                                       DataType_t const& /*data_type*/)
{
    DISTCONV_CHECK_MIOPEN(
        miopenInitConvolutionNdDescriptor(conv_desc,
                                          array_len,
                                          const_cast<int*>(pad),
                                          const_cast<int*>(stride),
                                          const_cast<int*>(dilation),
                                          mode));
}

inline void
copy_convolution_descriptor(miopenConvolutionDescriptor_t& dst,
                            miopenConvolutionDescriptor_t const& src)
{
    int spatial_dims = -1;
    // This gets the correct value for spatial_dims.
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(
        src, 0, &spatial_dims, nullptr, nullptr, nullptr, nullptr));

    std::vector<int> data;
    data.reserve(3 * spatial_dims);
    int* const pads = data.data();
    int* const strides = data.data() + spatial_dims;
    int* const dilations = data.data() + 2 * spatial_dims;
    miopenConvolutionMode_t mode;
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(
        src, spatial_dims, &spatial_dims, pads, strides, dilations, &mode));
    DISTCONV_CHECK_MIOPEN(miopenInitConvolutionNdDescriptor(
        dst, spatial_dims, pads, strides, dilations, mode));
}

inline constexpr auto default_conv_mode = miopenConvolution;

inline size_t
get_conv_forward_workspace_size(Handle_t const& /*handle*/,
                                TensorDescriptor_t const& /*in_desc*/,
                                FilterDescriptor_t const& /*filter_desc*/,
                                ConvolutionDescriptor_t const& /*conv_desc*/,
                                TensorDescriptor_t const& /*out_desc*/,
                                ConvFwdAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

inline size_t
get_conv_bwd_data_workspace_size(Handle_t const& /*handle*/,
                                 FilterDescriptor_t const& /*filter_desc*/,
                                 TensorDescriptor_t const& /*dy_desc*/,
                                 ConvolutionDescriptor_t const& /*conv_desc*/,
                                 TensorDescriptor_t const& /*dx_desc*/,
                                 ConvBwdDataAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

inline size_t
get_conv_bwd_filter_workspace_size(Handle_t const& /*handle*/,
                                   TensorDescriptor_t const& /*in_desc*/,
                                   TensorDescriptor_t const& /*dy_Desc*/,
                                   ConvolutionDescriptor_t const& /*conv_Desc*/,
                                   FilterDescriptor_t const& /*dw_desc*/,
                                   ConvBwdFilterAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

template <typename T>
inline void apply_fwd_bias(Handle_t handle,
                           T const& alpha,
                           TensorDescriptor_t const& bias_desc,
                           void const* const bias,
                           T const& beta,
                           TensorDescriptor_t const& y_desc,
                           void* const y)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionForwardBias(
        handle, &alpha, bias_desc, bias, &beta, y_desc, y));
}

template <typename T>
inline void apply_bwd_bias(Handle_t handle,
                           T const& alpha,
                           TensorDescriptor_t const& dy_desc,
                           void const* dy_data,
                           T const& beta,
                           TensorDescriptor_t const& db_desc,
                           void* const db_data)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardBias(
        handle, &alpha, dy_desc, dy_data, &beta, db_desc, db_data));
}

namespace details
{
inline miopenIndexType_t get_index_type()
{
    char const* env = std::getenv("H2_MIOPEN_POOLING_INDEX_SIZE");
    if (env)
    {
        int const bytes = std::atoi(env);
        switch (bytes)
        {
        case 8: return miopenIndexUint8;
        case 16: return miopenIndexUint16;
        case 32: return miopenIndexUint32;
        case 64: return miopenIndexUint64;
        }
    }
    return miopenIndexUint32;
}

inline void print_index_type(miopenPoolingDescriptor_t desc,
                             std::string const& delim = "**",
                             std::ostream& os = std::cerr)
{
    miopenIndexType_t idx_t;
    DISTCONV_CHECK_MIOPEN(miopenGetPoolingIndexType(desc, &idx_t));
    std::cerr << delim << " (" << desc << ") INDEX_TYPE = miopenIndexUint";
    switch (idx_t)
    {
    case miopenIndexUint8: std::cerr << 8; break;
    case miopenIndexUint16: std::cerr << 16; break;
    case miopenIndexUint32: std::cerr << 32; break;
    case miopenIndexUint64: std::cerr << 64; break;
    }
    std::cerr << " " << delim << std::endl;
}

} // namespace details

inline miopenPoolingDescriptor_t make_pooling_descriptor()
{
    miopenPoolingDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreatePoolingDescriptor(&desc));
    DISTCONV_CHECK_MIOPEN(
        miopenSetPoolingIndexType(desc, details::get_index_type()));
    return desc;
}

inline void destroy_pooling_descriptor(miopenPoolingDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyPoolingDescriptor(desc));
}

inline void setup_pooling_descriptor(miopenPoolingDescriptor_t& desc,
                                     miopenPoolingMode_t mode,
                                     int nb_dims,
                                     int* window_dim,
                                     int* pad,
                                     int* stride)
{
    DISTCONV_CHECK_MIOPEN(miopenSetNdPoolingDescriptor(
        desc, mode, nb_dims, window_dim, pad, stride));
    DISTCONV_CHECK_MIOPEN(
        miopenSetPoolingIndexType(desc, details::get_index_type()));
}

inline int get_pooling_descriptor_dims(miopenPoolingDescriptor_t const& desc)
{
    int num_dims = -1;
    DISTCONV_CHECK_MIOPEN(miopenGetNdPoolingDescriptor(
        desc, 0, nullptr, &num_dims, nullptr, nullptr, nullptr));
    return num_dims;
}

inline void copy_pooling_descriptor(miopenPoolingDescriptor_t& dst,
                                    miopenPoolingDescriptor_t const& src)
{
    int num_dims = get_pooling_descriptor_dims(src);
    miopenPoolingMode_t mode;
    miopenNanPropagation_t nan_prop;
    std::vector<int> data;
    data.reserve(3 * num_dims);
    int* const window_dims = data.data();
    int* const padding = data.data() + num_dims;
    int* const strides = data.data() + 2 * num_dims;
    DISTCONV_CHECK_MIOPEN(miopenGetNdPoolingDescriptor(
        src, num_dims, &mode, &num_dims, window_dims, padding, strides));
    DISTCONV_CHECK_MIOPEN(miopenSetNdPoolingDescriptor(
        dst, mode, num_dims, window_dims, padding, strides));

    miopenIndexType_t idx_t;
    DISTCONV_CHECK_MIOPEN(miopenGetPoolingIndexType(src, &idx_t));
    DISTCONV_CHECK_MIOPEN(miopenSetPoolingIndexType(dst, idx_t));
}

namespace details
{
void set_workspace(miopenPoolingDescriptor_t const& desc, void* workspace);
void* get_workspace(miopenPoolingDescriptor_t const& desc);
void clear_workspace(miopenPoolingDescriptor_t const& desc);
std::pair<void*, size_t> make_workspace(miopenHandle_t handle,
                                        miopenPoolingDescriptor_t desc,
                                        miopenTensorDescriptor_t out_desc);

} // namespace details

template <typename T>
inline void pooling_forward(miopenHandle_t handle,
                            miopenPoolingDescriptor_t desc,
                            T const& alpha,
                            miopenTensorDescriptor_t const& in_desc,
                            void const* in_data,
                            T const& beta,
                            miopenTensorDescriptor_t const& out_desc,
                            void* out_data,
                            bool training)
{
    // Set up the index type first.
    DISTCONV_CHECK_MIOPEN(
        miopenSetPoolingIndexType(desc, details::get_index_type()));
    // Then get the workspace size.
    auto workspace = (training ? details::make_workspace(handle, desc, out_desc)
                               : std::make_pair((void*) nullptr, (size_t) 0UL));
    DISTCONV_CHECK_MIOPEN(miopenPoolingForward(handle,
                                               desc,
                                               &alpha,
                                               in_desc,
                                               in_data,
                                               &beta,
                                               out_desc,
                                               out_data,
                                               /*do_backward=*/training,
                                               workspace.first,
                                               workspace.second));
}

template <typename T>
inline void pooling_backward(miopenHandle_t handle,
                             miopenPoolingDescriptor_t desc,
                             T const& alpha,
                             miopenTensorDescriptor_t const& out_desc,
                             void const* out_data,
                             miopenTensorDescriptor_t const& d_out_desc,
                             void const* d_out_data,
                             miopenTensorDescriptor_t const& in_desc,
                             void const* in_data,
                             T const& beta,
                             miopenTensorDescriptor_t const& d_in_desc,
                             void* d_in_data)
{
    // FIXME
    void* workspace = details::get_workspace(desc);
    assert_always((bool) workspace);
    DISTCONV_CHECK_MIOPEN(miopenPoolingBackward(handle,
                                                desc,
                                                &alpha,
                                                out_desc,
                                                out_data,
                                                d_out_desc,
                                                d_out_data,
                                                in_desc,
                                                in_data,
                                                &beta,
                                                d_in_desc,
                                                d_in_data,
                                                workspace));
    details::clear_workspace(desc);
}

inline miopenActivationDescriptor_t make_activation_descriptor()
{
    miopenActivationDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateActivationDescriptor(&desc));
    return desc;
}

inline void
destroy_activation_descriptor(miopenActivationDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyActivationDescriptor(desc));
}

inline void copy_activation_descriptor(miopenActivationDescriptor_t& dst,
                                       miopenActivationDescriptor_t const& src)
{
    miopenActivationMode_t mode;
    double alpha, beta, gamma;
    DISTCONV_CHECK_MIOPEN(
        miopenGetActivationDescriptor(src, &mode, &alpha, &beta, &gamma));
    DISTCONV_CHECK_MIOPEN(
        miopenSetActivationDescriptor(dst, mode, alpha, beta, gamma));
}

inline void setup_relu_activation_descriptor(miopenActivationDescriptor_t& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenSetActivationDescriptor(
        desc, miopenActivationRELU, 0.0, 0.0, 0.0));
}

template <typename T>
inline void activation_forward(miopenHandle_t handle,
                               miopenActivationDescriptor_t const& desc,
                               T const& alpha,
                               miopenTensorDescriptor_t const& in_desc,
                               void const* in_data,
                               T const& beta,
                               miopenTensorDescriptor_t const& out_desc,
                               void* out_data)
{
    DISTCONV_CHECK_MIOPEN(miopenActivationForward(
        handle, desc, &alpha, in_desc, in_data, &beta, out_desc, out_data));
}

template <typename T>
inline void activation_backward(miopenHandle_t handle,
                                miopenActivationDescriptor_t const& desc,
                                T const& alpha,
                                miopenTensorDescriptor_t const& out_desc,
                                void const* out_data,
                                miopenTensorDescriptor_t const& d_out_desc,
                                void const* d_out_data,
                                miopenTensorDescriptor_t const& in_desc,
                                void const* in_data,
                                T const& beta,
                                miopenTensorDescriptor_t const& d_in_desc,
                                void* d_in_data)
{
    DISTCONV_CHECK_MIOPEN(miopenActivationBackward(handle,
                                                   desc,
                                                   &alpha,
                                                   out_desc,
                                                   out_data,
                                                   d_out_desc,
                                                   d_out_data,
                                                   in_desc,
                                                   in_data,
                                                   &beta,
                                                   d_in_desc,
                                                   d_in_data));
}

struct Options
{
    bool m_overlap_halo_exchange = false;
    bool m_deterministic = false;
    bool m_enable_profiling = false;
    float m_ws_capacity_factor = 1.0;
    Options(bool overlap_halo_exchange = false,
            bool deterministic = false,
            bool enable_profiling = false,
            bool ws_capacity_factor = 1.0)
        : m_overlap_halo_exchange(overlap_halo_exchange),
          m_deterministic(deterministic),
          m_enable_profiling(enable_profiling),
          m_ws_capacity_factor(ws_capacity_factor)
    {
        set_by_environment_variables();
    }
    void set_by_environment_variables()
    {
        if (std::getenv("DISTCONV_OVERLAP_HALO_EXCHANGE"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DISTCONV_OVERLAP_HALO_EXCHANGE"
                                            << " detected";
            m_overlap_halo_exchange = true;
        }
        if (std::getenv("DISTCONV_DETERMINISTIC"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DISTCONV_DETERMINISTIC"
                                            << " detected";
            m_deterministic = true;
        }
        if (std::getenv("DISTCONV_ENABLE_PROFILING"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DISTCONV_ENABLE_PROFILING"
                                            << " detected";
            m_enable_profiling = true;
        }
        if (std::getenv("DISTCONV_WS_CAPACITY_FACTOR"))
        {
            util::MPIRootPrintStreamDebug() << "Environment variable: "
                                            << "DISTCONV_WS_CAPACITY_FACTOR"
                                            << " detected";
            m_ws_capacity_factor =
                atof(std::getenv("DISTCONV_WS_CAPACITY_FACTOR"));
        }
    }
};

// Backend context
class BackendMIOpen
{
public:
    BackendMIOpen(MPI_Comm comm,
                  miopenHandle_t miopen_h,
                  Options const& opts = Options())
        : m_miopen_h(miopen_h),
          m_stream(h2::gpu::make_stream()),
          m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
          m_p2p(comm),
#endif // DISTCONV_HAS_P2P
          m_opts(opts)
    {
        init(comm);
    }

    BackendMIOpen(MPI_Comm comm,
                  miopenHandle_t miopen_h,
                  hipStream_t stream,
                  Options const& opts = Options())
        : m_miopen_h(miopen_h),
          m_stream(stream),
          m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
          m_p2p(comm),
#endif // DISTCONV_HAS_P2P
          m_opts(opts)
    {
        init(comm);
    }

    virtual ~BackendMIOpen()
    {
#ifdef DISTCONV_HAS_P2P
        m_p2p.disconnect_all();
#endif // DISTCONV_HAS_P2P
    }

    std::string get_name() const { return std::string("MIOPEN"); }

    Options const& get_options() { return m_opts; }

    void wait() { h2::gpu::sync(m_stream); }

    MPI_Comm get_comm() { return m_comm; }

    std::shared_ptr<Al::NCCLBackend::comm_type> get_al_mpi_cuda_comm()
    {
        return m_al_mpi_cuda_comm;
    }

    Al::NCCLBackend::comm_type& get_al_nccl_comm() { return *m_al_nccl_comm; }

    miopenHandle_t get_handle() { return m_miopen_h; }

    hipStream_t get_stream() { return m_stream; }

    void ensure_workspace(size_t size)
    {
        // util::PrintStreamDebug() << "Requested Workspace: " << size << "\n";
        if (m_ws.get_size() < size)
            m_ws.allocate(size);
        // util::PrintStreamDebug() << "Workspace: " << size << "\n";
    }

    void* get_workspace(size_t size)
    {
        ensure_workspace(size);
        return m_ws.get();
    }

    void enable_nvtx_marking(bool b = true) { m_enable_nvtx = b; }

    void disable_nvtx_marking() { enable_nvtx_marking(false); }

    bool is_nvtx_enabled() const { return m_enable_nvtx; }

#ifdef DISTCONV_HAS_P2P
    p2p::P2P& get_p2p() { return m_p2p; }
#endif // DISTCONV_HAS_P2P

    hipStream_t get_internal_stream(int idx)
    {
        assert_always(idx < (int) m_internal_streams.size());
        return m_internal_streams[idx];
    }

    hipStream_t get_internal_stream_pr(int idx)
    {
        assert_always(idx < (int) m_internal_streams_pr.size());
        return m_internal_streams_pr[idx];
    }

    std::shared_ptr<Al::NCCLBackend::comm_type>&
    get_internal_al_mpi_cuda_comm(int idx)
    {
        assert_always(idx < (int) m_internal_streams_pr.size());
        return m_internal_al_mpi_cuda_comms[idx];
    }

    void wait_main_stream(int idx)
    {
        util::wait_stream(m_stream, get_internal_stream(idx));
    }

    void wait_main_stream_pr(int idx)
    {
        util::wait_stream(m_stream, get_internal_stream_pr(idx));
    }

    void wait_internal_stream(int idx)
    {
        util::wait_stream(get_internal_stream(idx), m_stream);
    }

    void wait_internal_stream_pr(int idx)
    {
        util::wait_stream(get_internal_stream_pr(idx), m_stream);
    }

    void sync_internal_stream(int idx)
    {
        util::sync_stream(m_stream, get_internal_stream(idx));
    }

    void sync_internal_stream_pr(int idx)
    {
        util::sync_stream(m_stream, get_internal_stream_pr(idx));
    }

    miopenConvFwdAlgorithm_t
    get_fwd_algorithm(std::string name,
                      miopenTensorDescriptor_t input_desc,
                      void const* input,
                      miopenTensorDescriptor_t filter_desc,
                      void const* filter,
                      miopenConvolutionDescriptor_t conv_desc,
                      miopenTensorDescriptor_t output_desc,
                      void* output,
                      size_t ws_size);

    miopenConvBwdDataAlgorithm_t
    get_bwd_data_algorithm(std::string name,
                           miopenTensorDescriptor_t filter_desc,
                           void const* filter,
                           miopenTensorDescriptor_t d_output_desc,
                           void const* d_output,
                           miopenConvolutionDescriptor_t conv_desc,
                           miopenTensorDescriptor_t d_input_desc,
                           void* d_input,
                           size_t ws_size);

    miopenConvBwdWeightsAlgorithm_t
    get_bwd_filter_algorithm(std::string name,
                             miopenTensorDescriptor_t input_desc,
                             void const* input,
                             miopenTensorDescriptor_t d_output_desc,
                             void const* d_output,
                             miopenConvolutionDescriptor_t conv_desc,
                             miopenTensorDescriptor_t d_filter_desc,
                             void* d_filter,
                             size_t ws_size);

    void init_chanfilt_channel_comm(index_t seg, MPI_Comm comm)
    {
        assert0(m_chanfilt_channel_comms.count(seg));
        m_chanfilt_channel_comms[seg] =
            std::unique_ptr<Al::NCCLBackend::comm_type>(
                new Al::NCCLBackend::comm_type(comm, get_stream()));
        util::MPIPrintStreamDebug()
            << "Setting up new chanfilt channel comm for segments=" << seg
            << " rank=" << m_chanfilt_channel_comms[seg]->rank() << " of "
            << m_chanfilt_channel_comms[seg]->size();
    }

    void init_chanfilt_filter_comm(index_t seg, MPI_Comm comm)
    {
        assert0(m_chanfilt_filter_comms.count(seg));
        m_chanfilt_filter_comms[seg] =
            std::unique_ptr<Al::NCCLBackend::comm_type>(
                new Al::NCCLBackend::comm_type(comm, get_stream()));
        util::MPIPrintStreamDebug()
            << "Setting up new chanfilt filter comm for segments=" << seg
            << " rank=" << m_chanfilt_filter_comms[seg]->rank() << " of "
            << m_chanfilt_filter_comms[seg]->size();
    }

    void init_segmented_ar_comm(index_t seg, MPI_Comm comm)
    {
        assert0(m_segmented_ar_comms.count(seg));
        util::MPIPrintStreamDebug()
            << "Setting up new segmented AR comm for segments=" << seg;
        m_segmented_ar_comms[seg] = std::unique_ptr<Al::NCCLBackend::comm_type>(
            new Al::NCCLBackend::comm_type(comm, get_stream()));
    }

    Al::NCCLBackend::comm_type* get_chanfilt_channel_comm(index_t seg)
    {
        if (m_chanfilt_channel_comms.count(seg) > 0)
        {
            return m_chanfilt_channel_comms[seg].get();
        }
        return nullptr;
    }

    Al::NCCLBackend::comm_type* get_chanfilt_filter_comm(index_t seg)
    {
        if (m_chanfilt_filter_comms.count(seg) > 0)
        {
            return m_chanfilt_filter_comms[seg].get();
        }
        return nullptr;
    }

    Al::NCCLBackend::comm_type* get_segmented_ar_comm(index_t seg)
    {
        if (m_segmented_ar_comms.count(seg) > 0)
        {
            return m_segmented_ar_comms[seg].get();
        }
        return nullptr;
    }

    inline Stream_t get_stream(miopenHandle_t handle)
    {
        Stream_t stream;
        DISTCONV_CHECK_MIOPEN(miopenGetStream(handle, &stream));
        return stream;
    }

    inline void set_stream(miopenHandle_t handle, Stream_t stream)
    {
        DISTCONV_CHECK_MIOPEN(miopenSetStream(handle, stream));
    }

    template <typename T>
    void convolution_forward(Handle_t handle,
                             T const& alpha,
                             TensorDescriptor_t const& in_desc,
                             void const* in_data,
                             FilterDescriptor_t const& filter_desc,
                             void const* filter_data,
                             ConvolutionDescriptor_t const& conv_desc,
                             ConvFwdAlgo_t const& conv_algo,
                             void* work_data,
                             size_t work_data_size,
                             T const& beta,
                             TensorDescriptor_t const& out_desc,
                             void* out_data)
    {
        DISTCONV_CHECK_MIOPEN(miopenConvolutionForward(handle,
                                                       &alpha,
                                                       in_desc,
                                                       in_data,
                                                       filter_desc,
                                                       filter_data,
                                                       conv_desc,
                                                       conv_algo,
                                                       &beta,
                                                       out_desc,
                                                       out_data,
                                                       work_data,
                                                       work_data_size));
    }

    template <typename T>
    void convolution_bwd_data(Handle_t handle,
                              T const& alpha,
                              FilterDescriptor_t const& filter_desc,
                              void const* filter_data,
                              TensorDescriptor_t const& dy_desc,
                              void const* dy_data,
                              ConvolutionDescriptor_t const& conv_desc,
                              ConvBwdDataAlgo_t const& conv_algo,
                              void* work_data,
                              size_t work_data_size,
                              T const& beta,
                              TensorDescriptor_t const& dx_desc,
                              void* dx_data)
    {
        DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardData(handle,
                                                            &alpha,
                                                            dy_desc,
                                                            dy_data,
                                                            filter_desc,
                                                            filter_data,
                                                            conv_desc,
                                                            conv_algo,
                                                            &beta,
                                                            dx_desc,
                                                            dx_data,
                                                            work_data,
                                                            work_data_size));
    }

    template <typename T>
    void convolution_bwd_filter(Handle_t handle,
                                T const& alpha,
                                TensorDescriptor_t const& in_desc,
                                void const* in_data,
                                TensorDescriptor_t const& dy_desc,
                                void const* dy_data,
                                ConvolutionDescriptor_t const& conv_desc,
                                ConvBwdFilterAlgo_t const& conv_algo,
                                void* work_data,
                                size_t work_data_size,
                                T const& beta,
                                FilterDescriptor_t const& dw_desc,
                                void* dw_data)
    {
        DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardWeights(handle,
                                                               &alpha,
                                                               dy_desc,
                                                               dy_data,
                                                               in_desc,
                                                               in_data,
                                                               conv_desc,
                                                               conv_algo,
                                                               &beta,
                                                               dw_desc,
                                                               dw_data,
                                                               work_data,
                                                               work_data_size));
    }

protected:
    MPI_Comm m_comm;
    std::shared_ptr<Al::NCCLBackend::comm_type> m_al_mpi_cuda_comm;
    // Keeps a heap object as copying a NCCLCommunicator destroys
    // ncclComm_t
    std::unique_ptr<Al::NCCLBackend::comm_type> m_al_nccl_comm;
    miopenHandle_t m_miopen_h;
    hipStream_t m_stream;
    tensor::Memory<tensor::CUDAAllocator> m_ws;
    bool m_enable_nvtx;
#ifdef DISTCONV_HAS_P2P
    p2p::P2P m_p2p;
#endif // DISTCONV_HAS_P2P
    // the number of internal streams; should be larger than the number
    // of bounary planes
    static constexpr int m_num_internal_streams = 8;
    std::vector<hipStream_t> m_internal_streams;
    static constexpr int m_num_internal_streams_pr = 8;
    std::vector<hipStream_t> m_internal_streams_pr;
    // The communicator of NCCLBackend creates new MPI communicators
    // when constructed even without no argument. Having them as heap
    // objects prevent that.
    std::vector<std::shared_ptr<Al::NCCLBackend::comm_type>>
        m_internal_al_mpi_cuda_comms;
    Options m_opts;

    // Segmented communicators for channel/filter communication.
    // Communicators for ranks within a single channel/filter domain with the
    // same channel indices on the filter tensor.
    std::unordered_map<index_t, std::unique_ptr<Al::NCCLBackend::comm_type>>
        m_chanfilt_channel_comms;
    // Same filter indices on the filter tensor.
    std::unordered_map<index_t, std::unique_ptr<Al::NCCLBackend::comm_type>>
        m_chanfilt_filter_comms;
    std::unordered_map<index_t, std::unique_ptr<Al::NCCLBackend::comm_type>>
        m_segmented_ar_comms;

    void init(MPI_Comm comm)
    {
        DISTCONV_CHECK_MPI(MPI_Comm_dup(comm, &m_comm));
        m_al_mpi_cuda_comm =
            std::make_shared<Al::NCCLBackend::comm_type>(m_comm,
                                                                 m_stream);
        m_al_nccl_comm.reset(new Al::NCCLBackend::comm_type(m_comm, m_stream));
        set_stream(m_miopen_h, m_stream);
        setup_internal_streams();
        setup_al_comms();
    }

    void setup_internal_streams()
    {
        for (int i = 0; i < m_num_internal_streams; ++i)
        {
            m_internal_streams.push_back(h2::gpu::make_stream_nonblocking());
        }
        for (int i = 0; i < m_num_internal_streams_pr; ++i)
        {
            m_internal_streams_pr.push_back(util::create_priority_stream());
        }
    }

    void setup_al_comms()
    {
        for (int i = 0; i < m_num_internal_streams_pr; ++i)
        {
            m_internal_al_mpi_cuda_comms.push_back(
                std::make_shared<Al::NCCLBackend::comm_type>(
                    m_comm, m_internal_streams_pr[i]));
        }
    }

    // miopenConvFwdAlgorithm_t get_fwd_algorithm_by_heuristics(
    //     miopenTensorDescriptor_t const& input_desc,
    //     miopenTensorDescriptor_t const& filter_desc,
    //     miopenConvolutionDescriptor_t const& conv_desc,
    //     miopenTensorDescriptor_t const& output_desc,
    //     size_t ws_size);

    // miopenConvFwdAlgorithm_t
    // autotune_fwd_algorithm(miopenTensorDescriptor_t const& input_desc,
    //                        void const* input,
    //                        miopenTensorDescriptor_t const& filter_desc,
    //                        void const* filter,
    //                        miopenConvolutionDescriptor_t const& conv_desc,
    //                        miopenTensorDescriptor_t const& output_desc,
    //                        void* output,
    //                        size_t ws_size);

    // miopenConvBwdDataAlgorithm_t get_bwd_data_algorithm_by_heuristics(
    //     miopenTensorDescriptor_t const& filter_desc,
    //     miopenTensorDescriptor_t const& d_output_desc,
    //     miopenConvolutionDescriptor_t const& conv_desc,
    //     miopenTensorDescriptor_t const& d_input_desc,
    //     size_t ws_size);

    // miopenConvBwdDataAlgorithm_t
    // autotune_bwd_data_algorithm(miopenTensorDescriptor_t const& filter_desc,
    //                             void const* filter,
    //                             miopenTensorDescriptor_t const&
    //                             d_output_desc, void const* d_output,
    //                             miopenConvolutionDescriptor_t const&
    //                             conv_desc, miopenTensorDescriptor_t const&
    //                             d_input_desc, void* d_input, size_t ws_size);

    // miopenConvBwdWeightsAlgorithm_t get_bwd_filter_algorithm_by_heuristics(
    //     miopenTensorDescriptor_t const& input_desc,
    //     miopenTensorDescriptor_t const& d_output_desc,
    //     miopenConvolutionDescriptor_t const& conv_desc,
    //     miopenTensorDescriptor_t const& d_filter_desc,
    //     size_t ws_size);

    // miopenConvBwdWeightsAlgorithm_t autotune_bwd_filter_algorithm(
    //     miopenTensorDescriptor_t const& input_desc,
    //     void const* input,
    //     miopenTensorDescriptor_t const& d_output_desc,
    //     void const* d_output,
    //     miopenConvolutionDescriptor_t const& conv_desc,
    //     miopenTensorDescriptor_t const& d_filter_desc,
    //     void* d_filter,
    //     size_t ws_size);
};

} // namespace miopen
} // namespace distconv
