#pragma once

#include "distconv/base.hpp"
#include "distconv/layers.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_mpi_rocm.hpp"
#include "distconv/tensor/tensor_rocm.hpp"
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

#include <miopen/miopen.h>

namespace distconv
{
namespace miopen
{

constexpr int nb_dims_requested = 100;

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
    assert_always(false && "FIXME: Need to recompute strides");
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

inline void
copy_convolution_descriptor(miopenConvolutionDescriptor_t& dst,
                            miopenConvolutionDescriptor_t const& src)
{
    int spatial_dims = -1;
    // This gets the correct value for spatial_dims.
    DISTCONV_CHECK_MIOPEN(
        miopenGetConvolutionNdDescriptor(
            src, 0, &spatial_dims, nullptr, nullptr, nullptr, nullptr));

    std::vector<int> data;
    data.reserve(3*spatial_dims);
    int* const pads = data.data();
    int* const strides = data.data() + spatial_dims;
    int* const dilations = data.data() + 2 * spatial_dims;
    miopenConvolutionMode_t mode;
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(src,
                                                         spatial_dims,
                                                         &spatial_dims,
                                                         pads,
                                                         strides,
                                                         dilations,
                                                         &mode));
    DISTCONV_CHECK_MIOPEN(miopenInitConvolutionNdDescriptor(
        dst, spatial_dims, pads, strides, dilations, mode));
}

inline int get_pooling_descriptor_dims(miopenPoolingDescriptor_t const& desc)
{
    int num_dims = -1;
    DISTCONV_CHECK_MIOPEN(miopenGetNdPoolingDescriptor(desc,
                                                       0,
                                                       nullptr,
                                                       &num_dims,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr));
    return num_dims;
}

inline void copy_pooling_descriptor(miopenPoolingDescriptor_t& dst,
                                    miopenPoolingDescriptor_t const& src)
{
    int num_dims = get_pooling_descriptor_dims(src);
    miopenPoolingMode_t mode;
    miopenNanPropagation_t nan_prop;
    std::vector<int> data;
    data.reserve(3*num_dims);
    int* const window_dims = data.data();
    int* const padding = data.data() + num_dims;
    int* const strides = data.data() + 2*num_dims;
    DISTCONV_CHECK_MIOPEN(miopenGetNdPoolingDescriptor(src,
                                                       num_dims,
                                                       &mode,
                                                       &num_dims,
                                                       window_dims,
                                                       padding,
                                                       strides));
    DISTCONV_CHECK_MIOPEN(miopenSetNdPoolingDescriptor(
        dst, mode, num_dims, window_dims, padding, strides));
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
          m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
          m_p2p(comm),
#endif // DISTCONV_HAS_P2P
          m_opts(opts)
    {
        DISTCONV_CHECK_HIP(hipStreamCreate(&m_stream));
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

    ~BackendMIOpen()
    {
#ifdef DISTCONV_HAS_P2P
        m_p2p.disconnect_all();
#endif // DISTCONV_HAS_P2P
    }

    std::string get_name() const
    {
        return std::string("MIOPEN");
    }

    Options const& get_options()
    {
        return m_opts;
    }

    void wait()
    {
        DISTCONV_CHECK_HIP(hipStreamSynchronize(m_stream));
    }

    MPI_Comm get_comm()
    {
        return m_comm;
    }

    std::shared_ptr<Al::HostTransferBackend::comm_type> get_al_mpi_hip_comm()
    {
        return m_al_mpi_hip_comm;
    }

    Al::NCCLBackend::comm_type& get_al_nccl_comm()
    {
        return *m_al_nccl_comm;
    }

    miopenHandle_t get_handle()
    {
        return m_miopen_h;
    }

    hipStream_t get_stream()
    {
        return m_stream;
    }

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

    void enable_nvtx_marking(bool b = true)
    {
        m_enable_nvtx = b;
    }

    void disable_nvtx_marking()
    {
        enable_nvtx_marking(false);
    }

    bool is_nvtx_enabled() const
    {
        return m_enable_nvtx;
    }

#ifdef DISTCONV_HAS_P2P
    p2p::P2P& get_p2p()
    {
        return m_p2p;
    }
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

    std::shared_ptr<Al::HostTransferBackend::comm_type>&
    get_internal_al_mpi_hip_comm(int idx)
    {
        assert_always(idx < (int) m_internal_streams_pr.size());
        return m_internal_al_mpi_hip_comms[idx];
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
    get_fwd_algorithm(std::string const& name,
                      miopenTensorDescriptor_t const* input_desc,
                      void const* input,
                      miopenTensorDescriptor_t const* filter_desc,
                      void const* filter,
                      miopenConvolutionDescriptor_t const* conv_desc,
                      miopenTensorDescriptor_t const* output_desc,
                      void* output,
                      size_t ws_size);

    miopenConvBwdDataAlgorithm_t
    get_bwd_data_algorithm(std::string const& name,
                           miopenTensorDescriptor_t const* filter_desc,
                           void const* filter,
                           miopenTensorDescriptor_t const* d_output_desc,
                           void const* d_output,
                           miopenConvolutionDescriptor_t const* conv_desc,
                           miopenTensorDescriptor_t const* d_input_desc,
                           void* d_input,
                           size_t ws_size);

    miopenConvBwdWeightsAlgorithm_t
    get_bwd_filter_algorithm(std::string const& name,
                             miopenTensorDescriptor_t const* input_desc,
                             void const* input,
                             miopenTensorDescriptor_t const* d_output_desc,
                             void const* d_output,
                             miopenConvolutionDescriptor_t const* conv_desc,
                             miopenTensorDescriptor_t const* d_filter_desc,
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

protected:
    MPI_Comm m_comm;
    std::shared_ptr<Al::HostTransferBackend::comm_type> m_al_mpi_hip_comm;
    // Keeps a heap object as copying a NCCLCommunicator destroys
    // ncclComm_t
    std::unique_ptr<Al::NCCLBackend::comm_type> m_al_nccl_comm;
    miopenHandle_t m_miopen_h;
    hipStream_t m_stream;
    tensor::Memory<tensor::HIPAllocator> m_ws;
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
    // The communicator of HostTransferBackend creates new MPI communicators
    // when constructed even without no argument. Having them as heap
    // objects prevent that.
    std::vector<std::shared_ptr<Al::HostTransferBackend::comm_type>>
        m_internal_al_mpi_hip_comms;
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
        m_al_mpi_hip_comm =
            std::make_shared<Al::HostTransferBackend::comm_type>(m_comm,
                                                                 m_stream);
        m_al_nccl_comm.reset(new Al::NCCLBackend::comm_type(m_comm, m_stream));
        DISTCONV_CHECK_MIOPEN(miopenSetStream(m_miopen_h, m_stream));
        setup_internal_streams();
        setup_al_comms();
    }

    void setup_internal_streams()
    {
        for (int i = 0; i < m_num_internal_streams; ++i)
        {
            hipStream_t s;
            DISTCONV_CHECK_HIP(
                hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
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
            m_internal_al_mpi_hip_comms.push_back(
                std::make_shared<Al::HostTransferBackend::comm_type>(
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
    //                             miopenTensorDescriptor_t const& d_output_desc,
    //                             void const* d_output,
    //                             miopenConvolutionDescriptor_t const& conv_desc,
    //                             miopenTensorDescriptor_t const& d_input_desc,
    //                             void* d_input,
    //                             size_t ws_size);

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

// #include "distconv/cudnn/convolution.hpp"
// #include "distconv/cudnn/pooling.hpp"
// #include "distconv/cudnn/relu.hpp"
// #include "distconv/cudnn/leaky_relu.hpp"
// #include "distconv/cudnn/batchnorm.hpp"
// #include "distconv/cudnn/softmax.hpp"
// #include "distconv/cudnn/cross_entropy.hpp"
// #include "distconv/cudnn/mean_squared_error.hpp"
