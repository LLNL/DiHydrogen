////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_cudnn.hpp"
#include "h2/gpu/memory_utils.hpp"

// NOTE: #include this AFTER the macro definitons of
// DISTCONV_ASSERT_PTR and DISTCONV_ASSERT_DEBUG.

// NOTE: #include this in the global namespace.

#include <cudnn.h>

#include <algorithm>     // std::find_if
#include <cstdlib>       // std::getenv, std::atoi
#include <string>        // std::string
#include <unordered_map> // std::unordered_map
#include <utility>       // std::pair
#include <vector>        // std::vector

namespace distconv
{
namespace
{

constexpr int nb_dims_requested = 100;

template <typename AlgoType, typename PerfType>
AlgoType find_best_algorithm(std::vector<PerfType> const& perf_results)
{
    std::map<AlgoType, float> time_map;
    for (const auto& res : perf_results)
    {
        assert_always(res.status == CUDNN_STATUS_SUCCESS);
        if (time_map.find(res.algo) == time_map.end())
        {
            time_map[res.algo] = 0;
        }
        time_map[res.algo] += res.time;
    }
    AlgoType best_algo = time_map.begin()->first;
    float min_time = std::numeric_limits<float>::max();
    for (const auto& x : time_map)
    {
        AlgoType algo = x.first;
        float time = x.second;
        if (time < min_time)
        {
            min_time = time;
            best_algo = algo;
        }
    }
    return best_algo;
}

} // namespace

// Runtime supplement

void GPUDNNBackend::record_event(Event_t event, Stream_t stream)
{
    DISTCONV_CHECK_CUDA(cudaEventRecord(event, stream));
}

float GPUDNNBackend::elapsed_time(Event_t start, Event_t end)
{
    float elapsed;
    DISTCONV_CHECK_CUDA(cudaEventElapsedTime(&elapsed, start, end));
    return elapsed;
}

size_t GPUDNNBackend::get_available_memory()
{
    size_t available, total;
    DISTCONV_CHECK_CUDA(cudaMemGetInfo(&available, &total));
    return available;
}

// Activation interface

auto GPUDNNBackend::make_activation_descriptor() -> ActivationDescriptor_t
{
    ActivationDescriptor_t desc;
    DISTCONV_CHECK_CUDNN(cudnnCreateActivationDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_activation_descriptor(
    ActivationDescriptor_t const& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroyActivationDescriptor(desc));
}

void GPUDNNBackend::copy_activation_descriptor(
    ActivationDescriptor_t& dst, ActivationDescriptor_t const& src)
{
    cudnnActivationMode_t mode;
    cudnnNanPropagation_t nan_prop;
    double coef;
    DISTCONV_CHECK_CUDNN(
        cudnnGetActivationDescriptor(src, &mode, &nan_prop, &coef));
    DISTCONV_CHECK_CUDNN(
        cudnnSetActivationDescriptor(dst, mode, nan_prop, coef));
}

void GPUDNNBackend::setup_relu_activation_descriptor(
    ActivationDescriptor_t& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnSetActivationDescriptor(
        desc, CUDNN_ACTIVATION_RELU, CUDNN_PROPAGATE_NAN, 0.0));
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
    DISTCONV_CHECK_CUDNN(cudnnActivationForward(
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
    DISTCONV_CHECK_CUDNN(cudnnActivationBackward(handle,
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
    DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_convolution_descriptor(
    ConvolutionDescriptor_t const& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(desc));
}

void GPUDNNBackend::set_convolution_group_count(
    ConvolutionDescriptor_t const& desc, int ngrps)
{
    DISTCONV_CHECK_CUDNN(cudnnSetConvolutionGroupCount(desc, ngrps));
}

void GPUDNNBackend::set_convolution_descriptor(
    ConvolutionDescriptor_t& conv_desc,
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

void GPUDNNBackend::copy_convolution_descriptor(
    ConvolutionDescriptor_t& dst, ConvolutionDescriptor_t const& src)
{
    int array_length;
    const int arrayLengthRequested = 100;
    int pads[arrayLengthRequested];
    int strides[arrayLengthRequested];
    int dilations[arrayLengthRequested];
    ConvolutionMode_t mode;
    DataType_t dt;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(src,
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
    DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(handle,
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
                                                 out_data));
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
    DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(handle,
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
                                                      dx_data));
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
    DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(handle,
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
                                                        dw_data));
}

size_t GPUDNNBackend::get_conv_forward_workspace_size(
    Handle_t const& handle,
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

size_t GPUDNNBackend::get_conv_bwd_data_workspace_size(
    Handle_t const& handle,
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

size_t GPUDNNBackend::get_conv_bwd_filter_workspace_size(
    Handle_t const& handle,
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

void GPUDNNBackend::apply_fwd_bias(Handle_t handle,
                                   void const* alpha,
                                   TensorDescriptor_t const& bias_desc,
                                   void const* const bias,
                                   void const* beta,
                                   TensorDescriptor_t const& y_desc,
                                   void* const y)
{
    DISTCONV_CHECK_CUDNN(
        cudnnAddTensor(handle, alpha, bias_desc, bias, beta, y_desc, y));
}

void GPUDNNBackend::apply_bwd_bias(Handle_t handle,
                                   void const* alpha,
                                   TensorDescriptor_t const& dy_desc,
                                   void const* dy_data,
                                   void const* beta,
                                   TensorDescriptor_t const& db_desc,
                                   void* const db_data)
{
    DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardBias(
        handle, alpha, dy_desc, dy_data, beta, db_desc, db_data));
}

auto GPUDNNBackend::get_fwd_algorithm_by_name(std::string const& name)
    -> ConvFwdAlgo_t
{
    return util::CUDNNConvolutionFwdAlgorithms::get_algo(name);
}

auto GPUDNNBackend::get_fwd_algorithm_by_heuristics(
    Handle_t handle,
    TensorDescriptor_t input_desc,
    FilterDescriptor_t filter_desc,
    ConvolutionDescriptor_t conv_desc,
    TensorDescriptor_t output_desc,
    size_t ws_size) -> ConvFwdAlgo_t
{
#if CUDNN_MAJOR < 8
    cudnnConvolutionFwdAlgo_t algo;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
        ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE,
        &algo));
    return algo;
#else  // CUDNN_MAJOR < 8
    int algo_count;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &algo_count));
    cudnnConvolutionFwdAlgoPerf_t* perf_results =
        new cudnnConvolutionFwdAlgoPerf_t[algo_count];
    int tested_algo_count = 0;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionForwardAlgorithm_v7(handle,
                                               input_desc,
                                               filter_desc,
                                               conv_desc,
                                               output_desc,
                                               algo_count,
                                               &tested_algo_count,
                                               perf_results));
    cudnnConvolutionFwdAlgo_t algo;
    for (int i = 0; i < tested_algo_count; i++)
    {
        if (perf_results[i].memory <= ws_size)
        {
            algo = perf_results[i].algo;
            delete[] perf_results;
            return algo;
        }
    }
    util::MPIPrintStreamError() << "No forward algorithm found for CUDNN";
    std::abort();
#endif // CUDNN_MAJOR < 8
}

auto GPUDNNBackend::get_fwd_algorithm_by_autotune(
    Handle_t handle,
    TensorDescriptor_t input_desc,
    void const* input,
    FilterDescriptor_t filter_desc,
    void const* filter,
    ConvolutionDescriptor_t conv_desc,
    TensorDescriptor_t output_desc,
    void* output,
    size_t ws_size) -> ConvFwdAlgo_t
{
    constexpr int trial_count = 5;
    constexpr int skip = 5;
    int algo_count;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionForwardAlgorithmMaxCount(handle, &algo_count));
    cudnnConvolutionFwdAlgoPerf_t* perf_results =
        new cudnnConvolutionFwdAlgoPerf_t[algo_count];
    std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    void* ws = nullptr;
    if (ws_size)
    {
        auto const stream = get_stream(handle);
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceAllocate(
            &ws, ws_size, stream));
    }
    for (int t = 0; t < trial_count + skip; ++t)
    {
        if (ws_size)
        {
            DISTCONV_CHECK_CUDNN(
                cudnnFindConvolutionForwardAlgorithmEx(handle,
                                                       input_desc,
                                                       input,
                                                       filter_desc,
                                                       filter,
                                                       conv_desc,
                                                       output_desc,
                                                       output,
                                                       algo_count,
                                                       &tested_algo_count,
                                                       perf_results,
                                                       ws,
                                                       ws_size));
        }
        else
        {
            DISTCONV_CHECK_CUDNN(
                cudnnFindConvolutionForwardAlgorithm(handle,
                                                     input_desc,
                                                     filter_desc,
                                                     conv_desc,
                                                     output_desc,
                                                     algo_count,
                                                     &tested_algo_count,
                                                     perf_results));
        }
        if (t > skip)
        {
            std::ostringstream ss;
            ss << "Forward autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                const auto& res = perf_results[i];
                ss << "("
                   << util::CUDNNConvolutionFwdAlgorithms::get_name(res.algo)
                   << ", ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    ss << res.time << " ms"
                       << ", " << res.memory / 1000 / 1000 << " KB";
                }
                else if (res.status == CUDNN_STATUS_ALLOC_FAILED)
                {
                    ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
                       << " MB required";
                }
                else
                {
                    ss << "INTERNAL ERROR";
                }
                ss << ") ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    perf_results_all.push_back(res);
                }
            }
            util::MPIPrintStreamDebug() << ss.str();
        }
    }
    delete[] perf_results;
    if (ws_size)
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceFree(ws));
    auto const best_algo =
        find_best_algorithm<cudnnConvolutionFwdAlgo_t,
                            cudnnConvolutionFwdAlgoPerf_t>(perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::CUDNNConvolutionFwdAlgorithms::get_name(best_algo);
    return best_algo;
}

auto GPUDNNBackend::get_bwd_data_algorithm_by_name(std::string const& name)
    -> ConvBwdDataAlgo_t
{
    return util::CUDNNConvolutionBwdDataAlgorithms::get_algo(name);
}

auto GPUDNNBackend::get_bwd_data_algorithm_by_heuristics(
    Handle_t handle,
    FilterDescriptor_t filter_desc,
    TensorDescriptor_t d_output_desc,
    ConvolutionDescriptor_t conv_desc,
    TensorDescriptor_t d_input_desc,
    size_t ws_size) -> ConvBwdDataAlgo_t
{
#if CUDNN_MAJOR < 8
    cudnnConvolutionBwdDataAlgo_t algo;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
        handle,
        filter_desc,
        d_output_desc,
        conv_desc,
        d_input_desc,
        CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE,
        &algo));
    return algo;
#else  // CUDNN_MAJOR < 8
    int algo_count;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &algo_count));
    cudnnConvolutionBwdDataAlgoPerf_t* perf_results =
        new cudnnConvolutionBwdDataAlgoPerf_t[algo_count];
    int tested_algo_count = 0;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(handle,
                                                    filter_desc,
                                                    d_output_desc,
                                                    conv_desc,
                                                    d_input_desc,
                                                    algo_count,
                                                    &tested_algo_count,
                                                    perf_results));
    cudnnConvolutionBwdDataAlgo_t algo;
    for (int i = 0; i < tested_algo_count; i++)
    {
        if (perf_results[i].memory <= ws_size)
        {
            algo = perf_results[i].algo;
            delete[] perf_results;
            return algo;
        }
    }
    util::MPIPrintStreamError() << "No backward data algorithm found for CUDNN";
    std::abort();
#endif // CUDNN_MAJOR < 8
}

auto GPUDNNBackend::get_bwd_data_algorithm_by_autotune(
    Handle_t handle,
    FilterDescriptor_t filter_desc,
    void const* filter,
    TensorDescriptor_t d_output_desc,
    void const* d_output,
    ConvolutionDescriptor_t conv_desc,
    TensorDescriptor_t d_input_desc,
    void* d_input,
    size_t ws_size) -> ConvBwdDataAlgo_t
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;
    int algo_count;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionBackwardDataAlgorithmMaxCount(handle, &algo_count));
    cudnnConvolutionBwdDataAlgoPerf_t* perf_results =
        new cudnnConvolutionBwdDataAlgoPerf_t[algo_count];
    std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    void* ws = nullptr;
    if (ws_size)
    {
        auto const stream = get_stream(handle);
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceAllocate(
            &ws, ws_size, stream));
    }
    for (int t = 0; t < trial_count + skip; ++t)
    {
        if (ws_size)
        {
            DISTCONV_CHECK_CUDNN(
                cudnnFindConvolutionBackwardDataAlgorithmEx(handle,
                                                            filter_desc,
                                                            filter,
                                                            d_output_desc,
                                                            d_output,
                                                            conv_desc,
                                                            d_input_desc,
                                                            d_input,
                                                            algo_count,
                                                            &tested_algo_count,
                                                            perf_results,
                                                            ws,
                                                            ws_size));
        }
        else
        {
            DISTCONV_CHECK_CUDNN(
                cudnnFindConvolutionBackwardDataAlgorithm(handle,
                                                          filter_desc,
                                                          d_output_desc,
                                                          conv_desc,
                                                          d_input_desc,
                                                          algo_count,
                                                          &tested_algo_count,
                                                          perf_results));
        }
        if (t > skip)
        {
            std::ostringstream ss;
            ss << "Backward data autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                const auto& res = perf_results[i];
                ss << "("
                   << util::CUDNNConvolutionBwdDataAlgorithms::get_name(
                          res.algo)
                   << ", ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    ss << res.time << " ms"
                       << ", " << res.memory / 1000 / 1000 << " MB";
                }
                else if (res.status == CUDNN_STATUS_ALLOC_FAILED)
                {
                    ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
                       << " MB required";
                }
                else
                {
                    ss << "INTERNAL ERROR";
                }
                ss << ") ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    perf_results_all.push_back(res);
                }
            }
            util::MPIPrintStreamDebug() << ss.str();
        }
    }
    delete[] perf_results;
    if (ws_size)
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceFree(ws));
    auto const best_algo =
        find_best_algorithm<cudnnConvolutionBwdDataAlgo_t,
                            cudnnConvolutionBwdDataAlgoPerf_t>(
            perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::CUDNNConvolutionBwdDataAlgorithms::get_name(best_algo);
    return best_algo;
}

auto GPUDNNBackend::get_bwd_filter_algorithm_by_name(std::string const& name)
    -> ConvBwdFilterAlgo_t
{
    return util::CUDNNConvolutionBwdFilterAlgorithms::get_algo(name);
}

auto GPUDNNBackend::get_bwd_filter_algorithm_by_heuristics(
    Handle_t handle,
    TensorDescriptor_t input_desc,
    TensorDescriptor_t d_output_desc,
    ConvolutionDescriptor_t conv_desc,
    FilterDescriptor_t d_filter_desc,
    size_t ws_size) -> ConvBwdFilterAlgo_t
{
#if CUDNN_MAJOR < 8
    cudnnConvolutionBwdFilterAlgo_t algo;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(
        handle,
        input_desc,
        d_output_desc,
        conv_desc,
        d_filter_desc,
        CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
        ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE,
        &algo));
    return algo;
#else  // CUDNN_MAJOR < 8
    int algo_count;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        handle, &algo_count));
    cudnnConvolutionBwdFilterAlgoPerf_t* perf_results =
        new cudnnConvolutionBwdFilterAlgoPerf_t[algo_count];
    int tested_algo_count = 0;
    DISTCONV_CHECK_CUDNN(
        cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle,
                                                      input_desc,
                                                      d_output_desc,
                                                      conv_desc,
                                                      d_filter_desc,
                                                      algo_count,
                                                      &tested_algo_count,
                                                      perf_results));
    cudnnConvolutionBwdFilterAlgo_t algo;
    for (int i = 0; i < tested_algo_count; i++)
    {
        if (perf_results[i].memory <= ws_size)
        {
            algo = perf_results[i].algo;
            delete[] perf_results;
            return algo;
        }
    }
    util::MPIPrintStreamError()
        << "No backward filter algorithm found for CUDNN";
    std::abort();
#endif // CUDNN_MAJOR < 8
}

auto GPUDNNBackend::get_bwd_filter_algorithm_by_autotune(
    Handle_t handle,
    TensorDescriptor_t input_desc,
    void const* input,
    TensorDescriptor_t d_output_desc,
    void const* d_output,
    ConvolutionDescriptor_t conv_desc,
    FilterDescriptor_t d_filter_desc,
    void* d_filter,
    size_t ws_size) -> ConvBwdFilterAlgo_t
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;
    int algo_count;
    DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
        handle, &algo_count));
    cudnnConvolutionBwdFilterAlgoPerf_t* perf_results =
        new cudnnConvolutionBwdFilterAlgoPerf_t[algo_count];
    std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    void* ws = nullptr;
    if (ws_size)
    {
        auto const stream = get_stream(handle);
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceAllocate(
            &ws, ws_size, stream));
    }
    for (int t = 0; t < trial_count + skip; ++t)
    {
        if (ws_size)
        {
            DISTCONV_CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
                handle,
                input_desc,
                input,
                d_output_desc,
                d_output,
                conv_desc,
                d_filter_desc,
                d_filter,
                algo_count,
                &tested_algo_count,
                perf_results,
                ws,
                ws_size));
        }
        else
        {
            DISTCONV_CHECK_CUDNN(
                cudnnFindConvolutionBackwardFilterAlgorithm(handle,
                                                            input_desc,
                                                            d_output_desc,
                                                            conv_desc,
                                                            d_filter_desc,
                                                            algo_count,
                                                            &tested_algo_count,
                                                            perf_results));
        }
        if (t > skip)
        {
            std::ostringstream ss;
            ss << "Backward filter autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                const auto& res = perf_results[i];

                ss << "("
                   << util::CUDNNConvolutionBwdFilterAlgorithms::get_name(
                          res.algo)
                   << ", ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    ss << res.time << " ms"
                       << ", " << res.memory / 1000 / 1000 << " MB";
                }
                else if (res.status == CUDNN_STATUS_ALLOC_FAILED)
                {
                    ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
                       << " MB required";
                }
                else
                {
                    ss << "INTERNAL ERROR";
                }
                ss << ") ";
                if (res.status == CUDNN_STATUS_SUCCESS)
                {
                    perf_results_all.push_back(res);
                }
            }
            util::MPIPrintStreamDebug() << ss.str();
        }
    }
    delete[] perf_results;
    if (ws_size)
        H2_CHECK_CUDA(h2::gpu::default_cub_allocator().DeviceFree(ws));
    auto const best_algo =
        find_best_algorithm<cudnnConvolutionBwdFilterAlgo_t,
                            cudnnConvolutionBwdFilterAlgoPerf_t>(
            perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::CUDNNConvolutionBwdFilterAlgorithms::get_name(best_algo);
    return best_algo;
}

// Handle interface

auto GPUDNNBackend::make_handle() -> Handle_t
{
    Handle_t handle;
    DISTCONV_CHECK_CUDNN(cudnnCreate(&handle));
    return handle;
}

void GPUDNNBackend::destroy_handle(Handle_t handle)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroy(handle));
}

auto GPUDNNBackend::get_stream(Handle_t handle) -> Stream_t
{
    Stream_t stream;
    DISTCONV_CHECK_CUDNN(cudnnGetStream(handle, &stream));
    return stream;
}

void GPUDNNBackend::set_stream(Handle_t handle, Stream_t stream)
{
    DISTCONV_CHECK_CUDNN(cudnnSetStream(handle, stream));
}

// Pooling interace

auto GPUDNNBackend::make_pooling_descriptor() -> PoolingDescriptor_t
{
    PoolingDescriptor_t desc;
    DISTCONV_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_pooling_descriptor(PoolingDescriptor_t const& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(desc));
}

void GPUDNNBackend::setup_pooling_descriptor(PoolingDescriptor_t& desc,
                                             PoolingMode_t mode,
                                             int nb_dims,
                                             int* window_dim,
                                             int* pad,
                                             int* stride)
{
    auto const max_pooling_nan_opt = CUDNN_PROPAGATE_NAN;
    DISTCONV_CHECK_CUDNN(cudnnSetPoolingNdDescriptor(
        desc, mode, max_pooling_nan_opt, nb_dims, window_dim, pad, stride));
}

void GPUDNNBackend::copy_pooling_descriptor(PoolingDescriptor_t& dst,
                                            PoolingDescriptor_t const& src)
{
    PoolingMode_t mode;
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

void GPUDNNBackend::pooling_forward(Handle_t handle,
                                    PoolingDescriptor_t desc,
                                    void const* alpha,
                                    TensorDescriptor_t const& in_desc,
                                    void const* in_data,
                                    void const* beta,
                                    TensorDescriptor_t const& out_desc,
                                    void* out_data,
                                    bool /*training*/)
{
    DISTCONV_CHECK_CUDNN(cudnnPoolingForward(
        handle, desc, alpha, in_desc, in_data, beta, out_desc, out_data));
}

void GPUDNNBackend::pooling_backward(Handle_t handle,
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
                                     void* d_in_data)
{
    cudnnStatus_t status = cudnnPoolingBackward(handle,
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

// Tensor interface

// template <typename Tensor>
// void GPUDNNBackend::setup_filter_descriptor(FilterDescriptor_t& desc,
//                                     Tensor const& tensor)
// {
//     // Lifted out of convolution.hpp; modified to not use data members.
//     DataType_t dt = util::get_cudnn_type<typename Tensor::data_type>();
//     const int_vector shape =
//         tensor.get_local_real_shape().template get_vector<int>();
//     DISTCONV_CHECK_CUDNN(
//         cudnnSetFilterNdDescriptor(desc,
//                                    dt,
//                                    CUDNN_TENSOR_NCHW,
//                                    shape.size(),
//                                    util::reverse(shape).data()));
// }

auto GPUDNNBackend::make_filter_descriptor() -> FilterDescriptor_t
{
    FilterDescriptor_t desc;
    DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_filter_descriptor(FilterDescriptor_t const& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroyFilterDescriptor(desc));
}

void GPUDNNBackend::set_filter_descriptor(FilterDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          size_t ndims,
                                          int const* dims)
{
    DISTCONV_CHECK_CUDNN(
        cudnnSetFilterNdDescriptor(desc, dt, CUDNN_TENSOR_NCHW, ndims, dims));
}

void GPUDNNBackend::set_filter_descriptor(FilterDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          std::vector<int> const& dims)
{
    set_filter_descriptor(desc, dt, dims.size(), dims.data());
}

auto GPUDNNBackend::make_tensor_descriptor() -> TensorDescriptor_t
{
    TensorDescriptor_t desc;
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&desc));
    return desc;
}

void GPUDNNBackend::destroy_tensor_descriptor(TensorDescriptor_t const& desc)
{
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(desc));
}

void GPUDNNBackend::set_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          size_t ndims,
                                          int* dims,
                                          int* strides)
{
    DISTCONV_CHECK_CUDNN(
        cudnnSetTensorNdDescriptor(desc, dt, ndims, dims, strides));
}

void GPUDNNBackend::set_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t const& dt,
                                          std::vector<int> const& dims,
                                          std::vector<int> const& strides)
{
    assert_eq(dims.size(), strides.size());
    DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        desc, dt, dims.size(), dims.data(), strides.data()));
}

int GPUDNNBackend::get_tensor_rank(TensorDescriptor_t const& desc)
{
    return get_tensor_num_dimensions(desc);
}

int GPUDNNBackend::get_tensor_dimension(TensorDescriptor_t const& desc, int d)
{
    DataType_t dt;
    int dims[nb_dims_requested];
    int strides[nb_dims_requested];
    int nbdims;
    DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
        desc, nb_dims_requested, &dt, &nbdims, dims, strides));
    d = d < 0 ? nbdims + d : d;
    assert_always(d < nbdims);
    return dims[nbdims - d - 1];
}

void GPUDNNBackend::set_tensor_dimension(TensorDescriptor_t& desc, int d, int n)
{
    DataType_t dt;
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

int GPUDNNBackend::get_tensor_num_dimensions(TensorDescriptor_t const& desc)
{
    DataType_t dt;
    int nbdims;
    DISTCONV_CHECK_CUDNN(
        cudnnGetTensorNdDescriptor(desc, 0, &dt, &nbdims, nullptr, nullptr));
    return nbdims;
}

void GPUDNNBackend::set_tensor_num_samples(TensorDescriptor_t& desc, int n)
{
    int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    set_tensor_dimension(desc, num_sample_dim, n);
}

int GPUDNNBackend::get_tensor_num_samples(TensorDescriptor_t const& desc)
{
    int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
    return get_tensor_dimension(desc, num_sample_dim);
}

void GPUDNNBackend::copy_tensor_descriptor(TensorDescriptor_t& dst,
                                           TensorDescriptor_t const& src)
{
    DataType_t dt;
    int dims[nb_dims_requested];
    int strides[nb_dims_requested];
    int nbdims;
    DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
        src, nb_dims_requested, &dt, &nbdims, dims, strides));
    DISTCONV_CHECK_CUDNN(
        cudnnSetTensorNdDescriptor(dst, dt, nbdims, dims, strides));
}

void GPUDNNBackend::get_tensor_descriptor(TensorDescriptor_t const& desc,
                                          DataType_t& dt,
                                          std::vector<int>& dims,
                                          std::vector<int>& strides)
{
    int rank;
    DISTCONV_CHECK_CUDNN(
        cudnnGetTensorNdDescriptor(desc, 0, &dt, &rank, nullptr, nullptr));

    dims.resize(rank);
    strides.resize(rank);
    DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
        desc, rank, &dt, &rank, dims.data(), strides.data()));
}

auto GPUDNNBackend::get_tensor_datatype(TensorDescriptor_t const& desc)
    -> DataType_t
{
    DataType_t dt;
    int rank;
    DISTCONV_CHECK_CUDNN(
        cudnnGetTensorNdDescriptor(desc, 0, &dt, &rank, nullptr, nullptr));
    return dt;
}

void GPUDNNBackend::copy_filter_descriptor(FilterDescriptor_t& dst,
                                           FilterDescriptor_t const& src)
{
    DataType_t dt;
    int dims[nb_dims_requested];
    int nbdims;
    cudnnTensorFormat_t fmt;
    DISTCONV_CHECK_CUDNN(cudnnGetFilterNdDescriptor(
        src, nb_dims_requested, &dt, &fmt, &nbdims, dims));
    DISTCONV_CHECK_CUDNN(
        cudnnSetFilterNdDescriptor(dst, dt, fmt, nbdims, dims));
}

int GPUDNNBackend::get_filter_descriptor_dimension(
    FilterDescriptor_t const& desc, int ND, int d)
{
    DataType_t dt;
    int dims[nb_dims_requested];
    int nbdims;
    cudnnTensorFormat_t fmt;
    DISTCONV_CHECK_CUDNN(
        cudnnGetFilterNdDescriptor(desc, ND, &dt, &fmt, &nbdims, dims));
    d = d < 0 ? nbdims + d : d;
    assert_always(d < nbdims);
    return dims[nbdims - d - 1];
}

} // namespace distconv
