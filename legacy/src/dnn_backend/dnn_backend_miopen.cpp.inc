#include "distconv/util/util_miopen.hpp"
#include "distconv/util/util_rocm.hpp"

// NOTE: #include this AFTER the macro definitons of
// DISTCONV_ASSERT_PTR and DISTCONV_ASSERT_DEBUG.

// NOTE: #include this in the global namespace.

#include <algorithm>     // std::find_if
#include <cstdlib>       // std::getenv, std::atoi
#include <string>        // std::string
#include <unordered_map> // std::unordered_map
#include <utility>       // std::pair
#include <vector>        // std::vector

#include <miopen/miopen.h>

// This is a bit extreme with the abort, but sure.
#define DISTCONV_CHECK_MIOPEN(miopen_call)                                     \
    do                                                                         \
    {                                                                          \
        miopenStatus_t const status_distconv_check_miopen = (miopen_call);     \
        if (status_distconv_check_miopen != miopenStatusSuccess)               \
        {                                                                      \
            ::distconv::util::PrintStreamError()                               \
                << "MIOpen error at " << __FILE__ << ":" << __LINE__ << ": "   \
                << miopenGetErrorString(status_distconv_check_miopen)          \
                << std::endl;                                                  \
            static_cast<void>(hipDeviceReset());                               \
            abort();                                                           \
        }                                                                      \
    } while (0)

namespace distconv
{
namespace
{

// FIXME (trb 07/25/2022): Need to setup "DEFAULT" and "DETERMINISTIC"
// for each of these.
struct MIOpenConvolutionFwdAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvFwdAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvFwdAlgorithm_t const algo) noexcept
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvFwdAlgorithm_t get_algo(std::string const& name) noexcept
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name) noexcept
    {
        return get_name(get_algo(name));
    }
};
MIOpenConvolutionFwdAlgorithms::map_type const
    MIOpenConvolutionFwdAlgorithms::algo_map = {
        {miopenConvolutionFwdAlgoGEMM, "GEMM"},
        {miopenConvolutionFwdAlgoDirect, "DIRECT"},
        {miopenConvolutionFwdAlgoFFT, "FFT"},
        {miopenConvolutionFwdAlgoWinograd, "WINOGRAD"},
        {miopenConvolutionFwdAlgoImplicitGEMM, "IMPLICIT_GEMM"},
};

struct MIOpenConvolutionBwdDataAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvBwdDataAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvBwdDataAlgorithm_t algo)
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvBwdDataAlgorithm_t get_algo(std::string const& name)
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name)
    {
        return get_name(get_algo(name));
    }
};
MIOpenConvolutionBwdDataAlgorithms::map_type const
    MIOpenConvolutionBwdDataAlgorithms::algo_map = {
        {miopenConvolutionBwdDataAlgoGEMM, "GEMM"},
        {miopenConvolutionBwdDataAlgoDirect, "DIRECT"},
        {miopenConvolutionBwdDataAlgoFFT, "FFT"},
        {miopenConvolutionBwdDataAlgoWinograd, "WINOGRAD"},
        {miopenTransposeBwdDataAlgoGEMM, "TRANSPOSE GEMM - DEPRECATED"},
        {miopenConvolutionBwdDataAlgoImplicitGEMM, "IMPLICIT_GEMM"},
};

struct MIOpenConvolutionBwdWeightsAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvBwdWeightsAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvBwdWeightsAlgorithm_t algo)
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvBwdWeightsAlgorithm_t get_algo(std::string const& name)
    {
        auto const iter =
            std::find_if(cbegin(algo_map),
                         cend(algo_map),
                         [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name)
    {
        return get_name(get_algo(name));
    }
};
MIOpenConvolutionBwdWeightsAlgorithms::map_type const
    MIOpenConvolutionBwdWeightsAlgorithms::algo_map = {
        {miopenConvolutionBwdWeightsAlgoGEMM, "GEMM"},
        {miopenConvolutionBwdWeightsAlgoDirect, "DIRECT"},
        {miopenConvolutionBwdWeightsAlgoWinograd, "WINOGRAD"},
        {miopenConvolutionBwdWeightsAlgoImplicitGEMM, "IMPLICIT_GEMM"},
};

} // namespace

// Runtime supplement

void GPUDNNBackend::record_event(Event_t event, Stream_t stream)
{
    H2_CHECK_HIP(hipEventRecord(event, stream));
}

float GPUDNNBackend::elapsed_time(Event_t start, Event_t stop)
{
    float elapsed_ms;
    H2_CHECK_HIP(hipEventElapsedTime(&elapsed_ms, start, stop));
    return elapsed_ms;
}

size_t GPUDNNBackend::get_available_memory()
{
    size_t free, total;
    H2_CHECK_HIP(hipMemGetInfo(&free, &total));
    return free;
}

// Activation interface

auto GPUDNNBackend::make_activation_descriptor() -> ActivationDescriptor_t
{
    ActivationDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateActivationDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_activation_descriptor(
    ActivationDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyActivationDescriptor(desc));
}

void GPUDNNBackend::copy_activation_descriptor(
    ActivationDescriptor_t& dst, ActivationDescriptor_t const& src)
{
    miopenActivationMode_t mode;
    double alpha, beta, gamma;
    DISTCONV_CHECK_MIOPEN(
        miopenGetActivationDescriptor(src, &mode, &alpha, &beta, &gamma));
    DISTCONV_CHECK_MIOPEN(
        miopenSetActivationDescriptor(dst, mode, alpha, beta, gamma));
}

void GPUDNNBackend::setup_relu_activation_descriptor(
    ActivationDescriptor_t& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenSetActivationDescriptor(
        desc, miopenActivationRELU, 0.0, 0.0, 0.0));
}

void GPUDNNBackend::activation_forward(Handle_t handle,
                                       ActivationDescriptor_t const& desc,
                                       void const* alpha,
                                       TensorDescriptor_t const& in_desc,
                                       void const* in_data,
                                       void const* beta,
                                       TensorDescriptor_t const& out_desc,
                                       void* out_data)
{
    DISTCONV_CHECK_MIOPEN(miopenActivationForward(
        handle, desc, alpha, in_desc, in_data, beta, out_desc, out_data));
}

void GPUDNNBackend::activation_backward(Handle_t handle,
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
                                        void* d_in_data)
{
    DISTCONV_CHECK_MIOPEN(miopenActivationBackward(handle,
                                                   desc,
                                                   alpha,
                                                   out_desc,
                                                   out_data,
                                                   d_out_desc,
                                                   d_out_data,
                                                   in_desc,
                                                   in_data,
                                                   beta,
                                                   d_in_desc,
                                                   d_in_data));
}

// Convolution interface

auto GPUDNNBackend::make_convolution_descriptor() -> ConvolutionDescriptor_t
{
    ConvolutionDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_convolution_descriptor(
    ConvolutionDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyConvolutionDescriptor(desc));
}

void GPUDNNBackend::set_convolution_group_count(
    ConvolutionDescriptor_t const& desc, int ngrps)
{
    DISTCONV_CHECK_MIOPEN(miopenSetConvolutionGroupCount(desc, ngrps));
}

void GPUDNNBackend::set_convolution_descriptor(
    ConvolutionDescriptor_t& conv_desc,
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

void GPUDNNBackend::copy_convolution_descriptor(
    ConvolutionDescriptor_t& dst, ConvolutionDescriptor_t const& src)
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

void GPUDNNBackend::convolution_forward(
    Handle_t handle,
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
    void* out_data)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionForward(handle,
                                                   alpha,
                                                   in_desc,
                                                   in_data,
                                                   filter_desc,
                                                   filter_data,
                                                   conv_desc,
                                                   conv_algo,
                                                   beta,
                                                   out_desc,
                                                   out_data,
                                                   work_data,
                                                   work_data_size));
}

void GPUDNNBackend::convolution_bwd_data(
    Handle_t handle,
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
    void* dx_data)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardData(handle,
                                                        alpha,
                                                        dy_desc,
                                                        dy_data,
                                                        filter_desc,
                                                        filter_data,
                                                        conv_desc,
                                                        conv_algo,
                                                        beta,
                                                        dx_desc,
                                                        dx_data,
                                                        work_data,
                                                        work_data_size));
}

void GPUDNNBackend::convolution_bwd_filter(
    Handle_t handle,
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
    void* dw_data)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardWeights(handle,
                                                           alpha,
                                                           dy_desc,
                                                           dy_data,
                                                           in_desc,
                                                           in_data,
                                                           conv_desc,
                                                           conv_algo,
                                                           beta,
                                                           dw_desc,
                                                           dw_data,
                                                           work_data,
                                                           work_data_size));
}

size_t GPUDNNBackend::get_conv_forward_workspace_size(
    Handle_t const& /*handle*/,
    TensorDescriptor_t const& /*in_desc*/,
    FilterDescriptor_t const& /*filter_desc*/,
    ConvolutionDescriptor_t const& /*conv_desc*/,
    TensorDescriptor_t const& /*out_desc*/,
    ConvFwdAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

size_t GPUDNNBackend::get_conv_bwd_data_workspace_size(
    Handle_t const& /*handle*/,
    FilterDescriptor_t const& /*filter_desc*/,
    TensorDescriptor_t const& /*dy_desc*/,
    ConvolutionDescriptor_t const& /*conv_desc*/,
    TensorDescriptor_t const& /*dx_desc*/,
    ConvBwdDataAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

size_t GPUDNNBackend::get_conv_bwd_filter_workspace_size(
    Handle_t const& /*handle*/,
    TensorDescriptor_t const& /*in_desc*/,
    TensorDescriptor_t const& /*dy_Desc*/,
    ConvolutionDescriptor_t const& /*conv_Desc*/,
    FilterDescriptor_t const& /*dw_desc*/,
    ConvBwdFilterAlgo_t const& /*algo*/)
{
    return 1 << 30;
}

void GPUDNNBackend::apply_fwd_bias(Handle_t handle,
                                   void const* alpha,
                                   TensorDescriptor_t const& bias_desc,
                                   void const* const bias,
                                   void const* beta,
                                   TensorDescriptor_t const& y_desc,
                                   void* const y)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionForwardBias(
        handle, alpha, bias_desc, bias, beta, y_desc, y));
}

void GPUDNNBackend::apply_bwd_bias(Handle_t handle,
                                   void const* alpha,
                                   TensorDescriptor_t const& dy_desc,
                                   void const* dy_data,
                                   void const* beta,
                                   TensorDescriptor_t const& db_desc,
                                   void* const db_data)
{
    DISTCONV_CHECK_MIOPEN(miopenConvolutionBackwardBias(
        handle, alpha, dy_desc, dy_data, beta, db_desc, db_data));
}

auto GPUDNNBackend::get_fwd_algorithm(std::string const& name,
                                      TensorDescriptor_t input_desc,
                                      void const* input,
                                      TensorDescriptor_t filter_desc,
                                      void const* filter,
                                      ConvolutionDescriptor_t conv_desc,
                                      TensorDescriptor_t output_desc,
                                      void* output,
                                      size_t ws_size) -> ConvFwdAlgo_t
{
    DISTCONV_ASSERT_PTR(nullptr);
    return static_cast<ConvFwdAlgo_t>(0);
}

auto GPUDNNBackend::get_bwd_data_algorithm(std::string const& name,
                                           TensorDescriptor_t filter_desc,
                                           void const* filter,
                                           TensorDescriptor_t d_output_desc,
                                           void const* d_output,
                                           ConvolutionDescriptor_t conv_desc,
                                           TensorDescriptor_t d_input_desc,
                                           void* d_input,
                                           size_t ws_size) -> ConvBwdDataAlgo_t
{
    DISTCONV_ASSERT_PTR(nullptr);
    return static_cast<ConvBwdDataAlgo_t>(0);
}

auto GPUDNNBackend::get_bwd_filter_algorithm(std::string const& name,
                                             TensorDescriptor_t input_desc,
                                             void const* input,
                                             TensorDescriptor_t d_output_desc,
                                             void const* d_output,
                                             ConvolutionDescriptor_t conv_desc,
                                             TensorDescriptor_t d_filter_desc,
                                             void* d_filter,
                                             size_t ws_size)
    -> ConvBwdFilterAlgo_t
{
    DISTCONV_ASSERT_PTR(nullptr);
    return static_cast<ConvBwdFilterAlgo_t>(0);
}

// Handle interface

std::string GPUDNNBackend::get_name()
{
    return "MIOpenBackend";
}

auto GPUDNNBackend::make_handle() -> Handle_t
{
    Handle_t handle;
    DISTCONV_CHECK_MIOPEN(miopenCreate(&handle));
    return handle;
}

void GPUDNNBackend::destroy_handle(Handle_t handle)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroy(handle));
}

auto GPUDNNBackend::get_stream(Handle_t handle) -> Stream_t
{
    Stream_t stream;
    DISTCONV_CHECK_MIOPEN(miopenGetStream(handle, &stream));
    return stream;
}

void GPUDNNBackend::set_stream(Handle_t handle, Stream_t stream)
{
    DISTCONV_CHECK_MIOPEN(miopenSetStream(handle, stream));
}

// Pooling interface

namespace
{

miopenIndexType_t get_index_type()
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

} // namespace

auto GPUDNNBackend::make_pooling_descriptor() -> PoolingDescriptor_t
{
    miopenPoolingDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreatePoolingDescriptor(&desc));
    DISTCONV_CHECK_MIOPEN(miopenSetPoolingIndexType(desc, get_index_type()));
    return desc;
}

void GPUDNNBackend::destroy_pooling_descriptor(PoolingDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyPoolingDescriptor(desc));
}

void GPUDNNBackend::setup_pooling_descriptor(PoolingDescriptor_t& desc,
                                             PoolingMode_t mode,
                                             int nb_dims,
                                             int* window_dim,
                                             int* pad,
                                             int* stride)
{
    DISTCONV_CHECK_MIOPEN(miopenSetNdPoolingDescriptor(
        desc, mode, nb_dims, window_dim, pad, stride));
    DISTCONV_CHECK_MIOPEN(miopenSetPoolingIndexType(desc, get_index_type()));
}

namespace
{

int get_pooling_descriptor_dims(miopenPoolingDescriptor_t const& desc)
{
    int num_dims = -1;
    DISTCONV_CHECK_MIOPEN(miopenGetNdPoolingDescriptor(
        desc, 0, nullptr, &num_dims, nullptr, nullptr, nullptr));
    return num_dims;
}

} // namespace

void GPUDNNBackend::copy_pooling_descriptor(
    miopenPoolingDescriptor_t& dst, miopenPoolingDescriptor_t const& src)
{
    int num_dims = get_pooling_descriptor_dims(src);
    miopenPoolingMode_t mode;
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

namespace
{
static std::unordered_map<miopenPoolingDescriptor_t, void*> workspace_map;

void set_workspace(miopenPoolingDescriptor_t const& desc, void* workspace)
{
    workspace_map[desc] = workspace;
}
void* get_workspace(miopenPoolingDescriptor_t const& desc)
{
    return workspace_map.at(desc);
}
void clear_workspace(miopenPoolingDescriptor_t const& desc)
{
    if (workspace_map.count(desc))
    {
        ::distconv::internal::RuntimeHIP::get_device_memory_pool().release(
            workspace_map[desc]);
        workspace_map.erase(desc);
    }
}
std::pair<void*, size_t> make_workspace(miopenHandle_t handle,
                                        miopenPoolingDescriptor_t desc,
                                        miopenTensorDescriptor_t out_desc)
{
    clear_workspace(desc);

    hipStream_t stream;
    DISTCONV_CHECK_MIOPEN(miopenGetStream(handle, &stream));

    size_t workspace_size = 0UL;
    DISTCONV_CHECK_MIOPEN(
        miopenPoolingGetWorkSpaceSizeV2(desc, out_desc, &workspace_size));
    void* workspace =
        ::distconv::internal::RuntimeHIP::get_device_memory_pool().get(
            workspace_size, stream);
    set_workspace(desc, workspace);
    return {workspace, workspace_size};
}

} // namespace

void GPUDNNBackend::pooling_forward(Handle_t handle,
                                    miopenPoolingDescriptor_t desc,
                                    void const* alpha,
                                    TensorDescriptor_t const& in_desc,
                                    void const* in_data,
                                    void const* beta,
                                    TensorDescriptor_t const& out_desc,
                                    void* out_data,
                                    bool training)
{
    // Set up the index type first.
    DISTCONV_CHECK_MIOPEN(miopenSetPoolingIndexType(desc, get_index_type()));
    // Then get the workspace size.
    auto workspace = (training ? make_workspace(handle, desc, out_desc)
                               : std::make_pair((void*) nullptr, (size_t) 0UL));
    DISTCONV_CHECK_MIOPEN(miopenPoolingForward(handle,
                                               desc,
                                               alpha,
                                               in_desc,
                                               in_data,
                                               beta,
                                               out_desc,
                                               out_data,
                                               /*do_backward=*/training,
                                               workspace.first,
                                               workspace.second));
}

void GPUDNNBackend::pooling_backward(Handle_t handle,
                                     miopenPoolingDescriptor_t desc,
                                     void const* alpha,
                                     TensorDescriptor_t const& out_desc,
                                     void const* out_data,
                                     TensorDescriptor_t const& d_out_desc,
                                     void const* d_out_data,
                                     TensorDescriptor_t const& in_desc,
                                     void const* in_data,
                                     void const* beta,
                                     TensorDescriptor_t const& d_in_desc,
                                     void* d_in_data)
{
    // FIXME
    void* workspace = get_workspace(desc);
    assert_always((bool) workspace);
    DISTCONV_CHECK_MIOPEN(miopenPoolingBackward(handle,
                                                desc,
                                                alpha,
                                                out_desc,
                                                out_data,
                                                d_out_desc,
                                                d_out_data,
                                                in_desc,
                                                in_data,
                                                beta,
                                                d_in_desc,
                                                d_in_data,
                                                workspace));
    clear_workspace(desc);
}

// Tensor interface

auto GPUDNNBackend::make_tensor_descriptor() -> TensorDescriptor_t
{
    TensorDescriptor_t desc;
    DISTCONV_CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_tensor_descriptor(TensorDescriptor_t const& desc)
{
    DISTCONV_CHECK_MIOPEN(miopenDestroyTensorDescriptor(desc));
}

void GPUDNNBackend::set_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          size_t ndims,
                                          int* dims,
                                          int* strides)
{
    DISTCONV_CHECK_MIOPEN(
        miopenSetTensorDescriptor(desc, dt, ndims, dims, strides));
}

void GPUDNNBackend::set_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          std::vector<int> const& dims,
                                          std::vector<int> const& strides)
{
    assert_eq(dims.size(), strides.size());
    set_tensor_descriptor(desc,
                          dt,
                          dims.size(),
                          const_cast<int*>(dims.data()),
                          const_cast<int*>(strides.data()));
}

auto GPUDNNBackend::make_filter_descriptor() -> TensorDescriptor_t
{
    return make_tensor_descriptor();
}

void GPUDNNBackend::destroy_filter_descriptor(TensorDescriptor_t const& desc)
{
    destroy_tensor_descriptor(desc);
}

void GPUDNNBackend::set_filter_descriptor(FilterDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          size_t ndims,
                                          int const* dims)
{
    set_filter_descriptor(desc, dt, std::vector<int>{dims, dims + ndims});
}

void GPUDNNBackend::set_filter_descriptor(FilterDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          std::vector<int> const& dims)
{
    auto const strides = get_fully_packed_strides(dims);
    set_tensor_descriptor(desc, dt, dims, strides);
}

int GPUDNNBackend::get_tensor_rank(TensorDescriptor_t const& desc)
{
    int num_dims = -1;
    DISTCONV_CHECK_MIOPEN(miopenGetTensorDescriptorSize(desc, &num_dims));
    return num_dims;
}

int GPUDNNBackend::get_tensor_dimension(TensorDescriptor_t const& desc, int d)
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

void GPUDNNBackend::set_tensor_dimension(TensorDescriptor_t& desc, int d, int n)
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

int GPUDNNBackend::get_tensor_num_dimensions(TensorDescriptor_t const& desc)
{
    return get_tensor_rank(desc);
}

void GPUDNNBackend::set_tensor_num_samples(TensorDescriptor_t& desc, int n)
{
    int const num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    set_tensor_dimension(desc, num_sample_dim, n);
}

int GPUDNNBackend::get_tensor_num_samples(TensorDescriptor_t const& desc)
{
    int const num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    return get_tensor_dimension(desc, num_sample_dim);
}

void GPUDNNBackend::copy_tensor_descriptor(TensorDescriptor_t& dst,
                                           TensorDescriptor_t const& src)
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

void GPUDNNBackend::get_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t& dt,
                                          std::vector<int>& dims,
                                          std::vector<int>& strides)
{
    auto const rank = get_tensor_rank(desc);
    dims.resize(rank);
    strides.resize(rank);
    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()));
}

auto GPUDNNBackend::get_tensor_datatype(TensorDescriptor_t const& desc)
    -> DataType_t
{
    DISTCONV_ASSERT_DEBUG(get_tensor_rank(desc) < 10);
    DataType_t dt;
    int dims[10];
    int strides[10];
    DISTCONV_CHECK_MIOPEN(miopenGetTensorDescriptor(desc, &dt, dims, strides));
    return dt;
}

void GPUDNNBackend::copy_filter_descriptor(TensorDescriptor_t& dst,
                                           TensorDescriptor_t const& src)
{
    copy_tensor_descriptor(dst, src);
}

auto GPUDNNBackend::get_filter_descriptor_dimension(
    TensorDescriptor_t const& desc, int /*ndims*/, int d) -> int
{
    return get_tensor_dimension(desc, d);
}

} // namespace distconv
