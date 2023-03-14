////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "distconv/base.hpp"
#include "distconv/layers.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_cudnn.hpp"

#ifdef DISTCONV_HAS_P2P
#include "p2p/p2p.hpp"
#endif // DISTCONV_HAS_P2P
#include <Al.hpp>
#include <cudnn.h>
#include <nvToolsExt.h>

#include <memory>
#include <string>
#include <type_traits>

#include <cuda_profiler_api.h>

namespace distconv
{
namespace cudnn
{
using ActivationDescriptor_t = cudnnActivationDescriptor_t;
using ConvolutionDescriptor_t = cudnnConvolutionDescriptor_t;
using ConvolutionMode_t = cudnnConvolutionMode_t;
using DataType_t = cudnnDataType_t;
using ConvFwdAlgo_t = cudnnConvolutionFwdAlgo_t;
using ConvBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo_t;
using ConvBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo_t;
using FilterDescriptor_t = cudnnFilterDescriptor_t;
using PoolingDescriptor_t = cudnnPoolingDescriptor_t;
using PoolingMode_t = cudnnPoolingMode_t;
using TensorDescriptor_t = cudnnTensorDescriptor_t;
using Handle_t = cudnnHandle_t;
using Stream_t = cudaStream_t;
using Event_t = cudaEvent_t;

inline constexpr auto default_conv_mode = CUDNN_CROSS_CORRELATION;
constexpr int nb_dims_requested = 100;

inline Handle_t make_handle()
{
    cudnnHandle_t handle;
    DISTCONV_CHECK_CUDNN(cudnnCreate(&handle));
    return handle;
}

inline void destroy_handle(cudnnHandle_t handle)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroy(handle));
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
class BackendCUDNN
{
public:
    BackendCUDNN(MPI_Comm comm,
                 cudnnHandle_t cudnn_h,
                 const Options& opts = Options())
        : m_cudnn_h(cudnn_h),
          m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
          m_p2p(comm),
#endif // DISTCONV_HAS_P2P
          m_opts(opts)
    {
        DISTCONV_CHECK_CUDA(cudaStreamCreate(&m_stream));
        init(comm);
    }

    BackendCUDNN(MPI_Comm comm,
                 cudnnHandle_t cudnn_h,
                 cudaStream_t stream,
                 const Options& opts = Options())
        : m_cudnn_h(cudnn_h),
          m_stream(stream),
          m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
          m_p2p(comm),
#endif // DISTCONV_HAS_P2P
          m_opts(opts)
    {
        init(comm);
    }

    virtual ~BackendCUDNN()
    {
#ifdef DISTCONV_HAS_P2P
        m_p2p.disconnect_all();
#endif // DISTCONV_HAS_P2P
    }

    std::string get_name() const { return std::string("CUDNN"); }

    const Options& get_options() { return m_opts; }

    void wait() { DISTCONV_CHECK_CUDA(cudaStreamSynchronize(m_stream)); }

    MPI_Comm get_comm() { return m_comm; }

    std::shared_ptr<Al::NCCLBackend::comm_type> get_al_mpi_cuda_comm()
    {
        return m_al_mpi_cuda_comm;
    }

    Al::NCCLBackend::comm_type& get_al_nccl_comm() { return *m_al_nccl_comm; }

    cudnnHandle_t get_handle() { return m_cudnn_h; }

    cudaStream_t get_stream() { return m_stream; }

    void ensure_workspace(size_t size)
    {
        // util::PrintStreamDebug() << "Requested Workspace: " << size << "\n";
        if (m_ws.get_size() < size)
        {
            m_ws.allocate(size);
        }
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

    cudaStream_t get_internal_stream(int idx)
    {
        assert_always(idx < (int) m_internal_streams.size());
        return m_internal_streams[idx];
    }

    cudaStream_t get_internal_stream_pr(int idx)
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

    cudnnConvolutionFwdAlgo_t
    get_fwd_algorithm(std::string name,
                      cudnnTensorDescriptor_t input_desc,
                      const void* input,
                      cudnnFilterDescriptor_t filter_desc,
                      const void* filter,
                      cudnnConvolutionDescriptor_t conv_desc,
                      cudnnTensorDescriptor_t output_desc,
                      void* output,
                      size_t ws_size);

    cudnnConvolutionBwdDataAlgo_t
    get_bwd_data_algorithm(std::string name,
                           cudnnFilterDescriptor_t filter_desc,
                           const void* filter,
                           cudnnTensorDescriptor_t d_output_desc,
                           const void* d_output,
                           cudnnConvolutionDescriptor_t conv_desc,
                           cudnnTensorDescriptor_t d_input_desc,
                           void* d_input,
                           size_t ws_size);

    cudnnConvolutionBwdFilterAlgo_t
    get_bwd_filter_algorithm(std::string name,
                             cudnnTensorDescriptor_t input_desc,
                             const void* input,
                             cudnnTensorDescriptor_t d_output_desc,
                             const void* d_output,
                             cudnnConvolutionDescriptor_t conv_desc,
                             cudnnFilterDescriptor_t d_filter_desc,
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

    inline Stream_t get_stream(cudnnHandle_t handle)
    {
        Stream_t stream;
        DISTCONV_CHECK_CUDNN(cudnnGetStream(handle, &stream));
        return stream;
    }

    inline void set_stream(cudnnHandle_t handle, Stream_t stream)
    {
        DISTCONV_CHECK_CUDNN(cudnnSetStream(handle, stream));
    }

    inline Event_t make_event()
    {
        cudaEvent_t event;
        DISTCONV_CHECK_CUDA(cudaEventCreate(&event));
        return event;
    }

    inline void destroy_event(cudaEvent_t const& event)
    {
        DISTCONV_CHECK_CUDA(cudaEventDestroy(event));
    }

    inline void record_event(cudaEvent_t const& event,
                             cudaStream_t const& stream)
    {
        DISTCONV_CHECK_CUDA(cudaEventRecord(event, stream));
    }

    inline float elapsed_time(cudaEvent_t const& start, cudaEvent_t const& end)
    {
        float elapsed;
        DISTCONV_CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, end));
        return elapsed;
    }

    inline size_t get_available_memory()
    {
        size_t available;
        size_t total;
        DISTCONV_CHECK_CUDA(cudaMemGetInfo(&available, &total));
        return available;
    }

    inline cudnnFilterDescriptor_t make_filter_descriptor()
    {
        cudnnFilterDescriptor_t desc;
        DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc));
        return desc;
    }

    inline void destroy_filter_descriptor(cudnnFilterDescriptor_t const& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnDestroyFilterDescriptor(desc));
    }

    inline cudnnTensorDescriptor_t make_tensor_descriptor()
    {
        cudnnTensorDescriptor_t desc;
        DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
        return desc;
    }

    inline void destroy_tensor_descriptor(cudnnTensorDescriptor_t const& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
    }

    template <typename T>
    inline void apply_fwd_bias(Handle_t handle,
                               T const& alpha,
                               TensorDescriptor_t const& bias_desc,
                               T const* const bias,
                               T const& beta,
                               TensorDescriptor_t const& y_desc,
                               T* const y)
    {
        DISTCONV_CHECK_CUDNN(
            cudnnAddTensor(handle, &alpha, bias_desc, bias, &beta, y_desc, y));
    }

    template <typename T>
    inline void apply_bwd_bias(Handle_t handle,
                               T const& alpha,
                               TensorDescriptor_t const& dy_desc,
                               T const* dy_data,
                               T const& beta,
                               TensorDescriptor_t const& db_desc,
                               T* const db_data)
    {
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardBias(
            handle, &alpha, dy_desc, dy_data, &beta, db_desc, db_data));
    }

    inline cudnnPoolingDescriptor_t make_pooling_descriptor()
    {
        cudnnPoolingDescriptor_t desc;
        DISTCONV_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc));
        return desc;
    }

    inline void destroy_pooling_descriptor(cudnnPoolingDescriptor_t const& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(desc));
    }

    inline void setup_pooling_descriptor(cudnnPoolingDescriptor_t& desc,
                                         cudnnPoolingMode_t mode,
                                         int nb_dims,
                                         int* window_dim,
                                         int* pad,
                                         int* stride)
    {
        auto const max_pooling_nan_opt = CUDNN_PROPAGATE_NAN;
        DISTCONV_CHECK_CUDNN(cudnnSetPoolingNdDescriptor(
            desc, mode, max_pooling_nan_opt, nb_dims, window_dim, pad, stride));
    }

    inline void copy_pooling_descriptor(cudnnPoolingDescriptor_t& dst,
                                        const cudnnPoolingDescriptor_t& src)
    {
        cudnnPoolingMode_t mode;
        cudnnNanPropagation_t nan_prop;
        int ndims;
        int window_dims[nb_dims_requested];
        int padding[nb_dims_requested];
        int strides[nb_dims_requested];

        DISTCONV_CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                         nb_dims_requested,
                                                         &mode,
                                                         &nan_prop,
                                                         &ndims,
                                                         window_dims,
                                                         padding,
                                                         strides));
        DISTCONV_CHECK_CUDNN(cudnnSetPoolingNdDescriptor(
            dst, mode, nan_prop, ndims, window_dims, padding, strides));
    }

    template <typename T>
    inline void pooling_forward(cudnnHandle_t handle,
                                cudnnPoolingDescriptor_t desc,
                                T const& alpha,
                                cudnnTensorDescriptor_t const& in_desc,
                                void const* in_data,
                                T const& beta,
                                cudnnTensorDescriptor_t const& out_desc,
                                void* out_data,
                                bool const& /*training*/)
    {
        DISTCONV_CHECK_CUDNN(cudnnPoolingForward(
            handle, desc, &alpha, in_desc, in_data, &beta, out_desc, out_data));
    }

    template <typename T>
    inline void pooling_backward(cudnnHandle_t handle,
                                 cudnnPoolingDescriptor_t desc,
                                 T const& alpha,
                                 cudnnTensorDescriptor_t const& out_desc,
                                 void const* out_data,
                                 cudnnTensorDescriptor_t const& d_out_desc,
                                 void const* d_out_data,
                                 cudnnTensorDescriptor_t const& in_desc,
                                 void const* in_data,
                                 T const& beta,
                                 cudnnTensorDescriptor_t const& d_in_desc,
                                 void* d_in_data)
    {
        cudnnStatus_t status = cudnnPoolingBackward(handle,
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
                                                    d_in_data);
        if (status != CUDNN_STATUS_SUCCESS)
        {
            util::MPIPrintStreamError()
                << "cuDNN error: " << cudnnGetErrorString(status) << "\n"
                << "Error at " << __FILE__ << ":" << __LINE__;
            if (status == CUDNN_STATUS_BAD_PARAM)
            {
                util::MPIPrintStreamError()
                    << "Parameters: "
                    << "output_d: " << out_desc << ", output: " << out_data
                    << ", d_output_d: " << d_out_desc
                    << ", d_output: " << d_out_data << ", input_d: " << in_desc
                    << ", input: " << in_data << ", d_input_d: " << d_in_desc
                    << ", d_input: " << d_in_data;
            }
            DISTCONV_CHECK_CUDA(cudaDeviceReset());
            abort();
        }
    }

    inline cudnnActivationDescriptor_t make_activation_descriptor()
    {
        cudnnActivationDescriptor_t desc;
        DISTCONV_CHECK_CUDNN(cudnnCreateActivationDescriptor(&desc));
        return desc;
    }

    inline void
    destroy_activation_descriptor(cudnnActivationDescriptor_t const& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnDestroyActivationDescriptor(desc));
    }

    inline void
    copy_activation_descriptor(cudnnActivationDescriptor_t& dst,
                               const cudnnActivationDescriptor_t& src)
    {
        cudnnActivationMode_t mode;
        cudnnNanPropagation_t nan_prop;
        double coef;
        DISTCONV_CHECK_CUDNN(
            cudnnGetActivationDescriptor(src, &mode, &nan_prop, &coef));
        DISTCONV_CHECK_CUDNN(
            cudnnSetActivationDescriptor(dst, mode, nan_prop, coef));
    }

    inline void
    setup_relu_activation_descriptor(cudnnActivationDescriptor_t& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnSetActivationDescriptor(
            desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
    }

    template <typename T>
    inline void activation_forward(cudnnHandle_t handle,
                                   cudnnActivationDescriptor_t const& desc,
                                   T const& alpha,
                                   cudnnTensorDescriptor_t const& in_desc,
                                   T const* in_data,
                                   T const& beta,
                                   cudnnTensorDescriptor_t const& out_desc,
                                   T* out_data)
    {
        DISTCONV_CHECK_CUDNN(cudnnActivationForward(
            handle, desc, &alpha, in_desc, in_data, &beta, out_desc, out_data));
    }

    template <typename T>
    inline void activation_backward(cudnnHandle_t handle,
                                    cudnnActivationDescriptor_t const& desc,
                                    T const& alpha,
                                    cudnnTensorDescriptor_t const& out_desc,
                                    T const* out_data,
                                    cudnnTensorDescriptor_t const& d_out_desc,
                                    T const* d_out_data,
                                    cudnnTensorDescriptor_t const& in_desc,
                                    T const* in_data,
                                    T const& beta,
                                    cudnnTensorDescriptor_t const& d_in_desc,
                                    T* d_in_data)
    {
        DISTCONV_CHECK_CUDNN(cudnnActivationBackward(handle,
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

    template <typename Tensor>
    inline void setup_filter_descriptor(FilterDescriptor_t& desc,
                                        Tensor const& tensor)
    {
        // Lifted out of convolution.hpp; modified to not use data members.
        cudnnDataType_t dt = util::get_cudnn_type<typename Tensor::data_type>();
        const int_vector shape =
            tensor.get_local_real_shape().template get_vector<int>();
        DISTCONV_CHECK_CUDNN(
            cudnnSetFilterNdDescriptor(desc,
                                       dt,
                                       CUDNN_TENSOR_NCHW,
                                       shape.size(),
                                       util::reverse(shape).data()));
    }

    template <typename Tensor, typename ShapeType>
    inline void setup_tensor_descriptor(cudnnTensorDescriptor_t& desc,
                                        const Tensor& tensor,
                                        const ShapeType& shape)
    {
        cudnnDataType_t dt = util::get_cudnn_type<typename Tensor::data_type>();
        assert_eq(tensor.get_num_dims(), shape.num_dims());

        if (shape.get_size() == 0)
            return;

        // set descriptor for input tensor
        // The size should include halo regions. Convolution will not be
        // done for the halo regions by disabling padding
        IndexVector strides = tensor::get_strides(tensor.get_local_shape(),
                                                  tensor.get_halo_width(),
                                                  tensor.get_pitch());

        util::MPIPrintStreamDebug()
            << "setup_tensor_descriptor. "
            << "tensor: " << tensor
            << ", shape: " << util::join_array(shape, ", ")
            << ", strides: " << util::join_array(strides, ", ") << "\n";

        DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
            desc,
            dt,
            shape.num_dims(),
            util::reverse(IntVector(shape)).data(),
            util::reverse(strides).get_vector<int>().data()));
    }

    template <typename Tensor>
    inline void setup_tensor_descriptor(cudnnTensorDescriptor_t& desc,
                                        const Tensor& tensor,
                                        const IntVector& halo_fwd,
                                        const IntVector& halo_bwd)
    {
        auto shape = tensor.get_local_shape();
        shape = shape + tensor::Shape(halo_fwd) + tensor::Shape(halo_bwd);
        return setup_tensor_descriptor(desc, tensor, shape);
    }

    template <typename Tensor>
    inline void
    setup_tensor_descriptor(cudnnTensorDescriptor_t& desc,
                            const Tensor& tensor,
                            const std::vector<bool>& include_halo_fwd,
                            const std::vector<bool>& include_halo_bwd)
    {
        const int nd = tensor.get_num_dims();
        auto overlap = tensor.get_overlap();
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
    inline void setup_tensor_descriptor(cudnnTensorDescriptor_t& desc,
                                        const Tensor& tensor,
                                        bool include_halo = true)
    {
        std::vector<bool> include_halo_array(tensor.get_num_dims(),
                                             include_halo);
        setup_tensor_descriptor(
            desc, tensor, include_halo_array, include_halo_array);
    }

    inline int get_tensor_dimension(const cudnnTensorDescriptor_t& desc, int d)
    {
        cudnnDataType_t dt;
        int dims[nb_dims_requested];
        int strides[nb_dims_requested];
        int nbdims;
        DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
            desc, nb_dims_requested, &dt, &nbdims, dims, strides));
        d = d < 0 ? nbdims + d : d;
        assert_always(d < nbdims);
        return dims[nbdims - d - 1];
    }

    inline void
    set_tensor_dimension(cudnnTensorDescriptor_t& desc, int d, int n)
    {
        cudnnDataType_t dt;
        int dims[nb_dims_requested];
        int strides[nb_dims_requested];
        int nbdims;
        DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
            desc, nb_dims_requested, &dt, &nbdims, dims, strides));
        d = d < 0 ? nbdims + d : d;
        assert_always(d < nbdims);
        dims[nbdims - d - 1] = n;
        DISTCONV_CHECK_CUDNN(
            cudnnSetTensorNdDescriptor(desc, dt, nbdims, dims, strides));
    }

    inline int get_tensor_num_dimensions(const cudnnTensorDescriptor_t& desc)
    {
        cudnnDataType_t dt;
        int nbdims;
        DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
            desc, 0, &dt, &nbdims, nullptr, nullptr));
        return nbdims;
    }

    inline void set_tensor_num_samples(cudnnTensorDescriptor_t& desc, int n)
    {
        int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
        set_tensor_dimension(desc, num_sample_dim, n);
    }

    inline int get_tensor_num_samples(const cudnnTensorDescriptor_t& desc)
    {
        int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
        return get_tensor_dimension(desc, num_sample_dim);
    }

    inline void copy_tensor_descriptor(cudnnTensorDescriptor_t& dst,
                                       const cudnnTensorDescriptor_t& src)
    {
        cudnnDataType_t dt;
        int dims[nb_dims_requested];
        int strides[nb_dims_requested];
        int nbdims;
        DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
            src, nb_dims_requested, &dt, &nbdims, dims, strides));
        DISTCONV_CHECK_CUDNN(
            cudnnSetTensorNdDescriptor(dst, dt, nbdims, dims, strides));
    }

    inline void copy_filter_descriptor(cudnnFilterDescriptor_t& dst,
                                       const cudnnFilterDescriptor_t& src)
    {
        cudnnDataType_t dt;
        int dims[nb_dims_requested];
        int nbdims;
        cudnnTensorFormat_t fmt;
        DISTCONV_CHECK_CUDNN(cudnnGetFilterNdDescriptor(
            src, nb_dims_requested, &dt, &fmt, &nbdims, dims));
        DISTCONV_CHECK_CUDNN(
            cudnnSetFilterNdDescriptor(dst, dt, fmt, nbdims, dims));
    }

    template <int ND>
    inline int
    get_filter_descriptor_dimension(const cudnnFilterDescriptor_t& desc, int d)
    {
        cudnnDataType_t dt;
        int dims[nb_dims_requested];
        int nbdims;
        cudnnTensorFormat_t fmt;
        DISTCONV_CHECK_CUDNN(
            cudnnGetFilterNdDescriptor(desc, ND, &dt, &fmt, &nbdims, dims));
        d = d < 0 ? nbdims + d : d;
        assert_always(d < nbdims);
        return dims[nbdims - d - 1];
    }

    inline cudnnConvolutionDescriptor_t make_convolution_descriptor()
    {
        cudnnConvolutionDescriptor_t desc;
        DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc));
        return desc;
    }

    inline void
    destroy_convolution_descriptor(cudnnConvolutionDescriptor_t const& desc)
    {
        DISTCONV_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(desc));
    }

    inline void
    set_convolution_group_count(cudnnConvolutionDescriptor_t const& desc,
                                int ngrps)
    {
        DISTCONV_CHECK_CUDNN(cudnnSetConvolutionGroupCount(desc, ngrps));
    }

    inline void set_convolution_descriptor(ConvolutionDescriptor_t& conv_desc,
                                           int const array_len,
                                           int const* const pad,
                                           int const* const stride,
                                           int const* const dilation,
                                           ConvolutionMode_t const& mode,
                                           DataType_t const& data_type)
    {
        DISTCONV_CHECK_CUDNN(
            cudnnSetConvolutionNdDescriptor(conv_desc,
                                            array_len,
                                            const_cast<int*>(pad),
                                            const_cast<int*>(stride),
                                            const_cast<int*>(dilation),
                                            mode,
                                            data_type));
    }

    inline void
    copy_convolution_descriptor(cudnnConvolutionDescriptor_t& dst,
                                const cudnnConvolutionDescriptor_t& src)
    {
        int array_length;
        const int arrayLengthRequested = 100;
        int pads[arrayLengthRequested];
        int strides[arrayLengthRequested];
        int dilations[arrayLengthRequested];
        cudnnConvolutionMode_t mode;
        cudnnDataType_t dt;
        DISTCONV_CHECK_CUDNN(
            cudnnGetConvolutionNdDescriptor(src,
                                            arrayLengthRequested,
                                            &array_length,
                                            pads,
                                            strides,
                                            dilations,
                                            &mode,
                                            &dt));
        DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
            dst, array_length, pads, strides, dilations, mode, dt));
    }

    inline size_t
    get_conv_forward_workspace_size(Handle_t const& handle,
                                    TensorDescriptor_t const& in_desc,
                                    FilterDescriptor_t const& filter_desc,
                                    ConvolutionDescriptor_t const& conv_desc,
                                    TensorDescriptor_t const& out_desc,
                                    ConvFwdAlgo_t const& algo)
    {
        size_t s;
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            handle, in_desc, filter_desc, conv_desc, out_desc, algo, &s));
        return s;
    }

    inline size_t
    get_conv_bwd_data_workspace_size(Handle_t const& handle,
                                     FilterDescriptor_t const& filter_desc,
                                     TensorDescriptor_t const& dy_desc,
                                     ConvolutionDescriptor_t const& conv_desc,
                                     TensorDescriptor_t const& dx_desc,
                                     ConvBwdDataAlgo_t const& algo)
    {
        size_t s;
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
            handle, filter_desc, dy_desc, conv_desc, dx_desc, algo, &s));
        return s;
    }

    inline size_t
    get_conv_bwd_filter_workspace_size(Handle_t const& handle,
                                       TensorDescriptor_t const& in_desc,
                                       TensorDescriptor_t const& dy_desc,
                                       ConvolutionDescriptor_t const& conv_desc,
                                       FilterDescriptor_t const& dw_desc,
                                       ConvBwdFilterAlgo_t const& algo)
    {
        size_t s;
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
            handle, in_desc, dy_desc, conv_desc, dw_desc, algo, &s));
        return s;
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
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(handle,
                                                     &alpha,
                                                     in_desc,
                                                     in_data,
                                                     filter_desc,
                                                     filter_data,
                                                     conv_desc,
                                                     conv_algo,
                                                     work_data,
                                                     work_data_size,
                                                     &beta,
                                                     out_desc,
                                                     out_data));
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
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
                                                          &alpha,
                                                          filter_desc,
                                                          filter_data,
                                                          dy_desc,
                                                          dy_data,
                                                          conv_desc,
                                                          conv_algo,
                                                          work_data,
                                                          work_data_size,
                                                          &beta,
                                                          dx_desc,
                                                          dx_data));
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
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle,
                                                            &alpha,
                                                            in_desc,
                                                            in_data,
                                                            dy_desc,
                                                            dy_data,
                                                            conv_desc,
                                                            conv_algo,
                                                            work_data,
                                                            work_data_size,
                                                            &beta,
                                                            dw_desc,
                                                            dw_data));
    }

protected:
    MPI_Comm m_comm;
    std::shared_ptr<Al::NCCLBackend::comm_type> m_al_mpi_cuda_comm;
    // Keeps a heap object as copying a NCCLCommunicator destroys
    // ncclComm_t
    std::unique_ptr<Al::NCCLBackend::comm_type> m_al_nccl_comm;
    cudnnHandle_t m_cudnn_h;
    cudaStream_t m_stream;
    tensor::Memory<tensor::CUDAAllocator> m_ws;
    bool m_enable_nvtx;
#ifdef DISTCONV_HAS_P2P
    p2p::P2P m_p2p;
#endif // DISTCONV_HAS_P2P
    // the number of internal streams; should be larger than the number
    // of bounary planes
    static constexpr int m_num_internal_streams = 8;
    std::vector<cudaStream_t> m_internal_streams;
    static constexpr int m_num_internal_streams_pr = 8;
    std::vector<cudaStream_t> m_internal_streams_pr;
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
        set_stream(m_cudnn_h, m_stream);
        setup_internal_streams();
        setup_al_comms();
    }

    void setup_internal_streams()
    {
        for (int i = 0; i < m_num_internal_streams; ++i)
        {
            cudaStream_t s;
            DISTCONV_CHECK_CUDA(
                cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
            m_internal_streams.push_back(s);
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

    cudnnConvolutionFwdAlgo_t get_fwd_algorithm_by_heuristics(
        const cudnnTensorDescriptor_t& input_desc,
        const cudnnFilterDescriptor_t& filter_desc,
        const cudnnConvolutionDescriptor_t& conv_desc,
        const cudnnTensorDescriptor_t& output_desc,
        size_t ws_size);

    cudnnConvolutionFwdAlgo_t
    autotune_fwd_algorithm(const cudnnTensorDescriptor_t& input_desc,
                           const void* input,
                           const cudnnFilterDescriptor_t& filter_desc,
                           const void* filter,
                           const cudnnConvolutionDescriptor_t& conv_desc,
                           const cudnnTensorDescriptor_t& output_desc,
                           void* output,
                           size_t ws_size);

    cudnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm_by_heuristics(
        const cudnnFilterDescriptor_t& filter_desc,
        const cudnnTensorDescriptor_t& d_output_desc,
        const cudnnConvolutionDescriptor_t& conv_desc,
        const cudnnTensorDescriptor_t& d_input_desc,
        size_t ws_size);

    cudnnConvolutionBwdDataAlgo_t
    autotune_bwd_data_algorithm(const cudnnFilterDescriptor_t& filter_desc,
                                const void* filter,
                                const cudnnTensorDescriptor_t& d_output_desc,
                                const void* d_output,
                                const cudnnConvolutionDescriptor_t& conv_desc,
                                const cudnnTensorDescriptor_t& d_input_desc,
                                void* d_input,
                                size_t ws_size);

    cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm_by_heuristics(
        const cudnnTensorDescriptor_t& input_desc,
        const cudnnTensorDescriptor_t& d_output_desc,
        const cudnnConvolutionDescriptor_t& conv_desc,
        const cudnnFilterDescriptor_t& d_filter_desc,
        size_t ws_size);

    cudnnConvolutionBwdFilterAlgo_t
    autotune_bwd_filter_algorithm(const cudnnTensorDescriptor_t& input_desc,
                                  const void* input,
                                  const cudnnTensorDescriptor_t& d_output_desc,
                                  const void* d_output,
                                  const cudnnConvolutionDescriptor_t& conv_desc,
                                  const cudnnFilterDescriptor_t& d_filter_desc,
                                  void* d_filter,
                                  size_t ws_size);
};

} // namespace cudnn
} // namespace distconv
