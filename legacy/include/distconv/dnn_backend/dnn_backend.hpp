////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "distconv_config.hpp"

#include "h2/gpu/runtime.hpp"

#ifdef DISTCONV_HAS_P2P
#include "p2p/p2p.hpp"
#endif // DISTCONV_HAS_P2P

#include <Al.hpp>
#include <mpi.h>

#if H2_HAS_CUDA

#define GPU_PROFILE_RANGE_POP nvtxRangePop
#define GPU_PROFILE_RANGE_PUSH nvtxRangePushA

#elif H2_HAS_ROCM

#define GPU_PROFILE_RANGE_POP roctxRangePop
#define GPU_PROFILE_RANGE_PUSH roctxRangePushA

#endif

#if H2_HAS_ROCM
#include <miopen/miopen.h>
#elif H2_HAS_CUDA
#include <cudnn.h>
#endif

namespace distconv
{

// Forward declarations
template <typename T>
class Vector;
using IntVector = Vector<int>;

struct Options
{
    bool overlap_halo_exchange;
    bool m_deterministic; // FIXME trb: LBANN compile hack
    bool enable_profiling;
    float ws_capacity_factor;

    Options(bool overlap_halo_exchange = false,
            bool deterministic = false,
            bool enable_profiling = false,
            float ws_capacity_factor = 1.0);
}; // struct Options

// Manage the collection of streams.
class StreamManager
{
public:
    /** @brief Construct StreamsManager
     *  @details For this version, a new stream (blocking) will be
     *           allocated to use as the "main stream". The second
     *           constructor will be used with this stream.
     *  @param[in] num_internal_streams The number of internal streams
     *             to use. This should be greater than the number of
     *             boundary planes.
     */
    StreamManager(size_t num_internal_streams);
    /** @brief Construct StreamsManager with a specific main stream.
     *  @details This will allocate two collections of
     *           `num_internal_streams`, one of which will be
     *           nonblocking streams and the other will be priority
     *           streams.
     *  @param[in] num_internal_streams The number of internal streams
     *             to use. This should be greater than the number of
     *             boundary planes.
     *  @param[in] main_stream The stream to use as the main stream.
     *             This object will manage the stream henceforth.
     */
    StreamManager(size_t num_internal_streams,
                  h2::gpu::DeviceStream main_stream);

    ~StreamManager() noexcept;

    /** @brief Get the number of available streams. */
    size_t num_streams() const noexcept;

    /** @brief The primary compute stream. */
    h2::gpu::DeviceStream stream() const noexcept;

    /** @brief Get an auxiliary stream.
     *  @param[in] idx Index of the stream to get.
     *  @throws std::out_of_range idx is greater than num_streams().
     */
    h2::gpu::DeviceStream internal_stream(size_t idx) const;

    /** @brief Get an auxiliary priority stream.
     *  @param[in] idx Index of the stream to get.
     *  @throws std::out_of_range idx is greater than num_streams().
     */
    h2::gpu::DeviceStream priority_stream(size_t idx) const;

private:
    using StreamContainer = std::vector<h2::gpu::DeviceStream>;
    h2::gpu::DeviceStream m_stream;
    StreamContainer m_internal_streams;
    StreamContainer m_priority_streams;
};

// Manage the required communicators. As far as I can tell, this is just a group
// of comms that basically get treated as globals. As usual in this file, they
// have nothing to do with MIOpen/cuDNN, they just need to be stashed somewhere.
class CommunicatorManager
{
public:
    using AlCommType = typename Al::NCCLBackend::comm_type;
    using AlInternalCommType = typename Al::NCCLBackend::comm_type;

public:
    CommunicatorManager(MPI_Comm comm, StreamManager const& stream_mgr);
    ~CommunicatorManager() = default;
    CommunicatorManager(CommunicatorManager const&) = delete;
    CommunicatorManager& operator=(CommunicatorManager const&) = delete;
    CommunicatorManager(CommunicatorManager&&) = delete;
    CommunicatorManager& operator=(CommunicatorManager&&) = delete;

    /** @name Primary communication context */
    ///@{

    MPI_Comm get_comm() const noexcept;
    AlCommType& get_al_nccl_comm();
#ifdef DISTCONV_HAS_P2P
    p2p::P2P& get_p2p()
    {
        return m_p2p;
    }
#endif // DISTCONV_HAS_P2P

    ///@}
    /** @name Halo exchange comms */
    ///@{

    size_t get_num_internal_comms() const noexcept;

    // API dictated by use -- refactoring shared_ptr out of
    // downstreams is probably not worth the effort.
    std::shared_ptr<AlInternalCommType> get_internal_comm(size_t idx) const;

    ///@}
    /** @name Channel/filter parallelism support */
    ///@{

    void init_segmented_ar_comm(size_t seg, MPI_Comm comm);
    void init_chanfilt_channel_comm(size_t seg, MPI_Comm comm);
    void init_chanfilt_filter_comm(size_t seg, MPI_Comm comm);

    AlCommType* get_segmented_ar_comm(size_t idx);
    AlCommType* get_chanfilt_channel_comm(size_t idx);
    AlCommType* get_chanfilt_filter_comm(size_t idx);

    ///@}

private:
#ifdef DISTCONV_HAS_P2P
    p2p::P2P m_p2p;
#endif // DISTCONV_HAS_P2P

    MPI_Comm m_comm;

    // These are stored on the heap for legacy reasons involving
    // copies. This class shouldn't be copied. So maybe this isn't
    // reasonable.
    std::unique_ptr<AlCommType> m_al_comm;
    std::vector<std::shared_ptr<AlInternalCommType>> m_internal_comms;

    // Segmented communicators for channel/filter communication.
    // Communicators for ranks within a single channel/filter domain with the
    // same channel indices on the filter tensor.
    using CommMap = std::unordered_map<size_t, AlInternalCommType>;
    CommMap m_segmented_ar_comms;
    CommMap m_chanfilt_channel_comms;
    // Same filter indices on the filter tensor.
    CommMap m_chanfilt_filter_comms;

}; // class CommunicatorManager

// This is essentially just a namespace on steroids (since you cannot
// template on a namespace or pass one around as an object).
/** @brief Type-ified repository for vendor-erased DNN ops.
 *
 *  cuDNN and MIOpen expose a C API (ugh), so the usual type-system
 *  gymnastics don't get us very far. This class exposes relevant
 *  functions with the `cudnn` or `miopen` namespace prefix removed.
 *  It also canonicalizes some API choices, generally landing on cuDNN
 *  to resolve differences between the vendor libraries.
 */
class GPUDNNBackend
{
public:
#if H2_HAS_ROCM
    using ActivationDescriptor_t = miopenActivationDescriptor_t;
    using ConvBwdDataAlgo_t = miopenConvBwdDataAlgorithm_t;
    using ConvBwdFilterAlgo_t = miopenConvBwdWeightsAlgorithm_t;
    using ConvFwdAlgo_t = miopenConvFwdAlgorithm_t;
    using ConvolutionDescriptor_t = miopenConvolutionDescriptor_t;
    using ConvolutionMode_t = miopenConvolutionMode_t;
    using DataType_t = miopenDataType_t;
    using Event_t = hipEvent_t;
    using FilterDescriptor_t = miopenTensorDescriptor_t;
    using Handle_t = miopenHandle_t;
    using PoolingDescriptor_t = miopenPoolingDescriptor_t;
    using PoolingMode_t = miopenPoolingMode_t;
    using Stream_t = hipStream_t;
    using TensorDescriptor_t = miopenTensorDescriptor_t;

    static constexpr auto default_conv_mode = miopenConvolution;
#elif H2_HAS_CUDA
    using ActivationDescriptor_t = cudnnActivationDescriptor_t;
    using ConvBwdDataAlgo_t = cudnnConvolutionBwdDataAlgo_t;
    using ConvBwdFilterAlgo_t = cudnnConvolutionBwdFilterAlgo_t;
    using ConvFwdAlgo_t = cudnnConvolutionFwdAlgo_t;
    using ConvolutionDescriptor_t = cudnnConvolutionDescriptor_t;
    using ConvolutionMode_t = cudnnConvolutionMode_t;
    using DataType_t = cudnnDataType_t;
    using Event_t = cudaEvent_t;
    using FilterDescriptor_t = cudnnFilterDescriptor_t;
    using Handle_t = cudnnHandle_t;
    using PoolingDescriptor_t = cudnnPoolingDescriptor_t;
    using PoolingMode_t = cudnnPoolingMode_t;
    using Stream_t = cudaStream_t;
    using TensorDescriptor_t = cudnnTensorDescriptor_t;

    static constexpr auto default_conv_mode = CUDNN_CROSS_CORRELATION;
#endif

public:
    /** @name GPU Runtime supplement */
    ///@{

    // These aren't in the h2::gpu:: runtime APIs
    static void record_event(Event_t event, Stream_t stream);
    static float elapsed_time(Event_t start, Event_t stop);
    static size_t get_available_memory();

    ///@}
    /** @name Activation interface */
    /// @{

    static ActivationDescriptor_t make_activation_descriptor();

    static void
    destroy_activation_descriptor(ActivationDescriptor_t const& desc);

    static void copy_activation_descriptor(ActivationDescriptor_t& dst,
                                           ActivationDescriptor_t const& src);
    static void setup_relu_activation_descriptor(ActivationDescriptor_t& desc);

    static void activation_forward(Handle_t handle,
                                   ActivationDescriptor_t const& desc,
                                   void const* alpha,
                                   TensorDescriptor_t const& in_desc,
                                   void const* in_data,
                                   void const* beta,
                                   TensorDescriptor_t const& out_desc,
                                   void* out_data);

    static void activation_backward(Handle_t handle,
                                    ActivationDescriptor_t const& desc,
                                    void const* alpha,
                                    TensorDescriptor_t const& out_desc,
                                    void const* out_data,
                                    TensorDescriptor_t const& d_out_desc,
                                    void const* d_out_data,
                                    TensorDescriptor_t const& in_desc,
                                    void const* in_data,
                                    void const* beta,
                                    TensorDescriptor_t const& d_in_desc,
                                    void* d_in_data);

    /// @}
    /** @name Convolution interface */
    /// @{

    static ConvolutionDescriptor_t make_convolution_descriptor();
    static void
    destroy_convolution_descriptor(ConvolutionDescriptor_t const& desc);

    static void set_convolution_group_count(ConvolutionDescriptor_t const& desc,
                                            int ngrps);
    static void set_convolution_descriptor(ConvolutionDescriptor_t& conv_desc,
                                           int const array_len,
                                           int const* const pad,
                                           int const* const stride,
                                           int const* const dilation,
                                           ConvolutionMode_t const& mode,
                                           DataType_t const& data_type);
    static void copy_convolution_descriptor(ConvolutionDescriptor_t& dst,
                                            ConvolutionDescriptor_t const& src);

    static void convolution_forward(Handle_t handle,
                                    void const* alpha,
                                    TensorDescriptor_t const& in_desc,
                                    void const* in_data,
                                    FilterDescriptor_t const& filter_desc,
                                    void const* filter_data,
                                    ConvolutionDescriptor_t const& conv_desc,
                                    ConvFwdAlgo_t const& conv_algo,
                                    void* work_data,
                                    size_t work_data_size,
                                    void const* beta,
                                    TensorDescriptor_t const& out_desc,
                                    void* out_data);

    static void convolution_bwd_data(Handle_t handle,
                                     void const* alpha,
                                     FilterDescriptor_t const& filter_desc,
                                     void const* filter_data,
                                     TensorDescriptor_t const& dy_desc,
                                     void const* dy_data,
                                     ConvolutionDescriptor_t const& conv_desc,
                                     ConvBwdDataAlgo_t const& conv_algo,
                                     void* work_data,
                                     size_t work_data_size,
                                     void const* beta,
                                     TensorDescriptor_t const& dx_desc,
                                     void* dx_data);

    static void convolution_bwd_filter(Handle_t handle,
                                       void const* alpha,
                                       TensorDescriptor_t const& in_desc,
                                       void const* in_data,
                                       TensorDescriptor_t const& dy_desc,
                                       void const* dy_data,
                                       ConvolutionDescriptor_t const& conv_desc,
                                       ConvBwdFilterAlgo_t const& conv_algo,
                                       void* work_data,
                                       size_t work_data_size,
                                       void const* beta,
                                       FilterDescriptor_t const& dw_desc,
                                       void* dw_data);

    static size_t
    get_conv_forward_workspace_size(Handle_t const& handle,
                                    TensorDescriptor_t const& in_desc,
                                    FilterDescriptor_t const& filter_desc,
                                    ConvolutionDescriptor_t const& conv_desc,
                                    TensorDescriptor_t const& out_desc,
                                    ConvFwdAlgo_t const& algo);

    static size_t
    get_conv_bwd_data_workspace_size(Handle_t const& handle,
                                     FilterDescriptor_t const& filter_desc,
                                     TensorDescriptor_t const& dy_desc,
                                     ConvolutionDescriptor_t const& conv_desc,
                                     TensorDescriptor_t const& dx_desc,
                                     ConvBwdDataAlgo_t const& algo);

    static size_t
    get_conv_bwd_filter_workspace_size(Handle_t const& handle,
                                       TensorDescriptor_t const& in_desc,
                                       TensorDescriptor_t const& dy_Desc,
                                       ConvolutionDescriptor_t const& conv_Desc,
                                       FilterDescriptor_t const& dw_desc,
                                       ConvBwdFilterAlgo_t const& algo);

    static void apply_fwd_bias(Handle_t handle,
                               void const* alpha,
                               TensorDescriptor_t const& bias_desc,
                               void const* const bias,
                               void const* beta,
                               TensorDescriptor_t const& y_desc,
                               void* y);

    static void apply_bwd_bias(Handle_t handle,
                               void const* alpha,
                               TensorDescriptor_t const& dy_desc,
                               void const* dy_data,
                               void const* beta,
                               TensorDescriptor_t const& db_desc,
                               void* db_data);

    static ConvFwdAlgo_t get_fwd_algorithm_by_name(std::string const& name);
    static ConvFwdAlgo_t
    get_fwd_algorithm_by_heuristics(Handle_t handle,
                                    TensorDescriptor_t input_desc,
                                    FilterDescriptor_t filter_desc,
                                    ConvolutionDescriptor_t conv_desc,
                                    TensorDescriptor_t output_desc,
                                    size_t ws_size);
    static ConvFwdAlgo_t
    get_fwd_algorithm_by_autotune(Handle_t handle,
                                  TensorDescriptor_t input_desc,
                                  void const* input,
                                  FilterDescriptor_t filter_desc,
                                  void const* filters,
                                  ConvolutionDescriptor_t conv_desc,
                                  TensorDescriptor_t output_desc,
                                  void* output,
                                  size_t ws_size);

    static ConvBwdDataAlgo_t
    get_bwd_data_algorithm_by_name(std::string const& name);
    static ConvBwdDataAlgo_t
    get_bwd_data_algorithm_by_heuristics(Handle_t handle,
                                         FilterDescriptor_t filter_desc,
                                         TensorDescriptor_t d_output_desc,
                                         ConvolutionDescriptor_t conv_desc,
                                         TensorDescriptor_t d_input_desc,
                                         size_t ws_size);
    static ConvBwdDataAlgo_t
    get_bwd_data_algorithm_by_autotune(Handle_t handle,
                                       FilterDescriptor_t filter_desc,
                                       void const* filter,
                                       TensorDescriptor_t d_output_desc,
                                       void const* d_output,
                                       ConvolutionDescriptor_t conv_desc,
                                       TensorDescriptor_t d_input_desc,
                                       void* d_input,
                                       size_t ws_size);

    static ConvBwdFilterAlgo_t
    get_bwd_filter_algorithm_by_name(std::string const& name);
    static ConvBwdFilterAlgo_t
    get_bwd_filter_algorithm_by_heuristics(Handle_t handle,
                                           TensorDescriptor_t input_desc,
                                           TensorDescriptor_t d_output_desc,
                                           ConvolutionDescriptor_t conv_desc,
                                           FilterDescriptor_t d_filter_desc,
                                           size_t ws_size);
    static ConvBwdFilterAlgo_t
    get_bwd_filter_algorithm_by_autotune(Handle_t handle,
                                         TensorDescriptor_t input_desc,
                                         void const* input,
                                         TensorDescriptor_t d_output_desc,
                                         void const* d_output,
                                         ConvolutionDescriptor_t conv_desc,
                                         FilterDescriptor_t d_filter_desc,
                                         void* d_filter,
                                         size_t ws_size);

    /// @}
    /** @name Handle interface */
    /// @{

    static Handle_t make_handle();

    static void destroy_handle(Handle_t handle);

    static void set_stream(Handle_t handle, Stream_t stream);

    static Stream_t get_stream(Handle_t handle);

    /// @}
    /** @name Pooling interface */
    /// @{

    static PoolingDescriptor_t make_pooling_descriptor();

    static void destroy_pooling_descriptor(PoolingDescriptor_t const& desc);

    static void setup_pooling_descriptor(PoolingDescriptor_t& desc,
                                         PoolingMode_t mode,
                                         int nb_dims,
                                         int* window_dim,
                                         int* pad,
                                         int* stride);

    static void copy_pooling_descriptor(PoolingDescriptor_t& dst,
                                        PoolingDescriptor_t const& src);

    static void pooling_forward(Handle_t handle,
                                PoolingDescriptor_t desc,
                                void const* alpha,
                                TensorDescriptor_t const& in_desc,
                                void const* in_data,
                                void const* beta,
                                TensorDescriptor_t const& out_desc,
                                void* out_data,
                                bool training);

    static void pooling_backward(Handle_t handle,
                                 PoolingDescriptor_t desc,
                                 void const* alpha,
                                 TensorDescriptor_t const& out_desc,
                                 void const* out_data,
                                 TensorDescriptor_t const& d_out_desc,
                                 void const* d_out_data,
                                 TensorDescriptor_t const& in_desc,
                                 void const* in_data,
                                 void const* beta,
                                 TensorDescriptor_t const& d_in_desc,
                                 void* d_in_data);

    /// @}
    /** @name Tensor interface */
    /// @{

    static FilterDescriptor_t make_filter_descriptor();

    static void destroy_filter_descriptor(FilterDescriptor_t const& desc);

    static void set_filter_descriptor(FilterDescriptor_t const& desc,
                                      DataType_t const& dt,
                                      size_t ndims,
                                      int const* dims);

    static void set_filter_descriptor(FilterDescriptor_t const& desc,
                                      DataType_t const& dt,
                                      std::vector<int> const& dims);

    static TensorDescriptor_t make_tensor_descriptor();

    static void destroy_tensor_descriptor(TensorDescriptor_t const& desc);

    static void set_tensor_descriptor(TensorDescriptor_t const& desc,
                                      DataType_t const& dt,
                                      size_t ndims,
                                      int* dims,
                                      int* strides);

    static void set_tensor_descriptor(TensorDescriptor_t const& desc,
                                      DataType_t const& dt,
                                      std::vector<int> const& dims,
                                      std::vector<int> const& strides);

    static int get_tensor_rank(TensorDescriptor_t const& desc);

    static int get_tensor_dimension(TensorDescriptor_t const& desc, int d);

    static void set_tensor_dimension(TensorDescriptor_t& desc, int d, int n);

    static int get_tensor_num_dimensions(TensorDescriptor_t const& desc);

    static void set_tensor_num_samples(TensorDescriptor_t& desc, int n);

    static int get_tensor_num_samples(TensorDescriptor_t const& desc);

    static void copy_tensor_descriptor(TensorDescriptor_t& dst,
                                       TensorDescriptor_t const& src);

    static void get_tensor_descriptor(TensorDescriptor_t const& desc,
                                      DataType_t& dt,
                                      std::vector<int>& dims,
                                      std::vector<int>& strides);

    static DataType_t get_tensor_datatype(TensorDescriptor_t const& desc);

    static void copy_filter_descriptor(FilterDescriptor_t& dst,
                                       FilterDescriptor_t const& src);

    static int get_filter_descriptor_dimension(FilterDescriptor_t const& desc,
                                               int ndims,
                                               int d);

    template <typename Tensor>
    static void setup_filter_descriptor(FilterDescriptor_t& desc,
                                        Tensor const& tensor);

    template <typename Tensor, typename ShapeType>
    static void setup_tensor_descriptor(TensorDescriptor_t& desc,
                                        Tensor const& tensor,
                                        ShapeType const& shape);

    template <typename Tensor>
    static void setup_tensor_descriptor(TensorDescriptor_t& desc,
                                        Tensor const& tensor,
                                        IntVector const& halo_fwd,
                                        IntVector const& halo_bwd);

    template <typename Tensor>
    static void
    setup_tensor_descriptor(TensorDescriptor_t& desc,
                            Tensor const& tensor,
                            std::vector<bool> const& include_halo_fwd,
                            std::vector<bool> const& include_halo_bwd);

    template <typename Tensor>
    static void setup_tensor_descriptor(TensorDescriptor_t& desc,
                                        Tensor const& tensor,
                                        bool include_halo = true);
    /// @}
}; // class MIOpenBackend

class Workspace
{
    size_t m_capacity;
    size_t m_size;
    void* m_ptr;

public:
    Workspace(size_t size) { realloc(size); }
    size_t capacity() const noexcept { return m_capacity; }
    size_t size() const noexcept { return m_size; }
    void* ptr() const noexcept { return m_ptr; }
    /** @brief Reallocate to a pointer of at least size bytes.
     *  @details Unlike C23, setting a size of zero will deallocate.
     */
    void* realloc(size_t size);
};

// The rest of the stuff.
//
// This interface will be defined in terms of types available via the
// VendorBackend. This would allow us, e.g., to potentially construct
// cuDNN, MIOpen, and OneDNN backends simultaneously, assuming the
// compiler issues were worked out and such heterogeneous hardware
// were available.
template <typename VendorBackendT>
class DNNBackend
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

    using AlCommType = CommunicatorManager::AlCommType;
    using InternalCommType = CommunicatorManager::AlInternalCommType;

public:
    DNNBackend(MPI_Comm comm, Handle_t handle, Options opts = Options{});
    DNNBackend(MPI_Comm comm,
               Handle_t handle,
               Stream_t stream,
               Options opts = Options{});

    /** @name Option queries. */
    ///@{

    bool deterministic() const noexcept { return m_opts.m_deterministic; }
    bool profiling() const noexcept { return m_opts.enable_profiling; }
    bool overlap_halo_exchange() const noexcept
    {
        return m_opts.overlap_halo_exchange;
    }
    float ws_capacity_factor() const noexcept
    {
        return m_opts.ws_capacity_factor;
    };

    ///@}
    /** @name Communicator accessors. */
    ///@{

    MPI_Comm get_comm() const noexcept;
    AlCommType& get_al_comm();
    AlCommType& get_al_nccl_comm() { return get_al_comm(); }
#ifdef DISTCONV_HAS_P2P
    p2p::P2P& get_p2p()
    {
        return m_comms.get_p2p();
    }
#endif // DISTCONV_HAS_P2P

    /** @brief HACK to help LBANN
     *
     *  @warning DO NOT USE THIS FUNCTION. It is a hack to avoid changes to
     * LBANN in a certain PR.
     *  @todo REMOVE THIS FUNCTION.
     */
    std::shared_ptr<AlCommType> get_al_mpi_cuda_comm() const
    {
        throw std::runtime_error("Don't use this function.");
        return nullptr;
    }

    size_t get_num_internal_comms() const noexcept;

    std::shared_ptr<InternalCommType> get_internal_comm(size_t idx) const;

    AlCommType* get_segmented_ar_comm(size_t idx);
    AlCommType* get_chanfilt_channel_comm(size_t idx);
    AlCommType* get_chanfilt_filter_comm(size_t idx);

    void init_chanfilt_channel_comm(size_t seg, MPI_Comm comm);
    void init_chanfilt_filter_comm(size_t seg, MPI_Comm comm);
    void init_segmented_ar_comm(size_t seg, MPI_Comm comm);

    ///@}
    /** @name Library management */
    ///@{

    Handle_t get_handle() const noexcept;
    /** @brief Get the stream this backend was constructed with. */
    Stream_t get_stream() const noexcept;
    Stream_t get_internal_priority_stream(size_t idx) const;
    void wait() const;

    ///@}
    /** @name Activation Operation */
    ///@{

    void activation_forward(ActivationDescriptor_t const& act_desc,
                            double alpha,
                            TensorDescriptor_t const& xdesc,
                            void const* x,
                            double beta,
                            TensorDescriptor_t const& ydesc,
                            void* y) const;

    virtual void activation_forward(ActivationDescriptor_t const& act_desc,
                                    double alpha,
                                    TensorDescriptor_t const& xdesc,
                                    void const* x,
                                    double beta,
                                    TensorDescriptor_t const& ydesc,
                                    void* y,
                                    Stream_t stream) const;

    void activation_backward(ActivationDescriptor_t const& act_desc,
                             double alpha,
                             TensorDescriptor_t const& ydesc,
                             void const* y,
                             TensorDescriptor_t const& dydesc,
                             void const* dy,
                             TensorDescriptor_t const& xdesc,
                             void const* x,
                             double beta,
                             TensorDescriptor_t const& dxdesc,
                             void* dx) const;

    virtual void activation_backward(ActivationDescriptor_t const& act_desc,
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
                                     Stream_t s) const;

    ///@}
    /** @name Convolution Operation */
    ///@{

    virtual size_t
    get_conv_forward_workspace_size(TensorDescriptor_t const& in_desc,
                                    FilterDescriptor_t const& filter_desc,
                                    ConvolutionDescriptor_t const& conv_desc,
                                    TensorDescriptor_t const& out_desc,
                                    ConvFwdAlgo_t const& algo) const;

    virtual size_t
    get_conv_bwd_data_workspace_size(FilterDescriptor_t const& filter_desc,
                                     TensorDescriptor_t const& dy_desc,
                                     ConvolutionDescriptor_t const& conv_desc,
                                     TensorDescriptor_t const& dx_desc,
                                     ConvBwdDataAlgo_t const& algo) const;

    virtual size_t
    get_conv_bwd_filter_workspace_size(TensorDescriptor_t const& in_desc,
                                       TensorDescriptor_t const& dy_Desc,
                                       ConvolutionDescriptor_t const& conv_Desc,
                                       FilterDescriptor_t const& dw_desc,
                                       ConvBwdFilterAlgo_t const& algo) const;

    virtual ConvFwdAlgo_t
    get_fwd_algorithm(std::string const& name,
                      TensorDescriptor_t const& input_desc,
                      void const* input,
                      FilterDescriptor_t const& filter_desc,
                      void const* filter,
                      ConvolutionDescriptor_t const& conv_desc,
                      TensorDescriptor_t const& output_desc,
                      void* output,
                      size_t ws_size) const;

    virtual ConvBwdDataAlgo_t
    get_bwd_data_algorithm(std::string const& name,
                           FilterDescriptor_t const& filter_desc,
                           void const* filter,
                           TensorDescriptor_t const& d_output_desc,
                           void const* d_output,
                           ConvolutionDescriptor_t const& conv_desc,
                           TensorDescriptor_t const& d_input_desc,
                           void* d_input,
                           size_t ws_size) const;

    virtual ConvBwdFilterAlgo_t
    get_bwd_filter_algorithm(std::string const& name,
                             TensorDescriptor_t const& input_desc,
                             void const* input,
                             TensorDescriptor_t const& d_output_desc,
                             void const* d_output,
                             ConvolutionDescriptor_t const& conv_desc,
                             FilterDescriptor_t const& d_filter_desc,
                             void* d_filter,
                             size_t ws_size) const;

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
                             void* y) const;

    virtual void convolution_forward(double alpha,
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
                                     Stream_t s) const;

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
                              void* dx_data) const;

    virtual void convolution_bwd_data(double alpha,
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
                                      Stream_t s) const;

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
                                void* dw_data) const;

    virtual void
    convolution_bwd_filter(double alpha,
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
                           Stream_t s) const;

    virtual void apply_fwd_bias(double alpha,
                                TensorDescriptor_t const& bias_desc,
                                void const* const bias,
                                double beta,
                                TensorDescriptor_t const& y_desc,
                                void* y);

    virtual void apply_bwd_bias(double alpha,
                                TensorDescriptor_t const& dy_desc,
                                void const* dy_data,
                                double beta,
                                TensorDescriptor_t const& db_desc,
                                void* db_data);

    ///@}
    /** @name Pooling Operation */
    ///@{
    void pooling_forward(PoolingDescriptor_t const& pooling_desc,
                         double alpha,
                         TensorDescriptor_t const& xdesc,
                         void const* x,
                         double beta,
                         TensorDescriptor_t const& ydesc,
                         void* y,
                         bool training) const;

    virtual void pooling_forward(PoolingDescriptor_t const& pooling_desc,
                                 double alpha,
                                 TensorDescriptor_t const& xdesc,
                                 void const* x,
                                 double beta,
                                 TensorDescriptor_t const& ydesc,
                                 void* y,
                                 bool training,
                                 Stream_t s) const;

    void pooling_backward(PoolingDescriptor_t const& pooling_desc,
                          double alpha,
                          TensorDescriptor_t const& ydesc,
                          void const* y,
                          TensorDescriptor_t const& dydesc,
                          void const* dy,
                          TensorDescriptor_t const& xdesc,
                          void const* x,
                          double beta,
                          TensorDescriptor_t const& dxdesc,
                          void* dx) const;

    virtual void pooling_backward(PoolingDescriptor_t const& pooling_desc,
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
                                  Stream_t s) const;
    ///@}

private:
    Handle_t m_handle;
    Options m_opts;
    StreamManager m_stream_mgr;
    CommunicatorManager m_comms;
}; // class DNNBackend

} // namespace distconv
