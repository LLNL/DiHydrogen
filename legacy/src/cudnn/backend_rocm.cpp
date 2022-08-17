#include "distconv/cudnn/backend_miopen.hpp"
#include "distconv/util/util_miopen.hpp"
#include "distconv/util/util_mpi.hpp"

#include <limits>
#include <string>

namespace distconv
{
namespace miopen
{
namespace
{
struct WSBuffer
{
    WSBuffer(size_t bytes)
        : m_buffer{internal::RuntimeHIP::get_device_memory_pool().get(bytes, 0)}
    {
        assert_always((bool) m_buffer);
    }
    ~WSBuffer()
    {
        if (m_buffer)
            internal::RuntimeHIP::get_device_memory_pool().release(m_buffer);
    }
    operator void*() { return m_buffer; }
    operator void const*() const { return m_buffer; }
    void* m_buffer = nullptr;
};

// Default workspace wize
constexpr size_t CONVOLUTION_WORKSPACE_SIZE = 1 << 30;

template <typename AlgoType>
struct PerfAlgo;

template <>
struct PerfAlgo<miopenConvFwdAlgorithm_t>
{
    static miopenConvFwdAlgorithm_t get(miopenConvAlgoPerf_t const& perf)
    {
        return perf.fwd_algo;
    }
};
template <>
struct PerfAlgo<miopenConvBwdWeightsAlgorithm_t>
{
    static miopenConvBwdWeightsAlgorithm_t get(miopenConvAlgoPerf_t const& perf)
    {
        return perf.bwd_weights_algo;
    }
};
template <>
struct PerfAlgo<miopenConvBwdDataAlgorithm_t>
{
    static miopenConvBwdDataAlgorithm_t get(miopenConvAlgoPerf_t const& perf)
    {
        return perf.bwd_data_algo;
    }
};

} // namespace

template <typename AlgoType>
static AlgoType get_algo(miopenConvAlgoPerf_t const& perf)
{
    return PerfAlgo<AlgoType>::get(perf);
}

static miopenConvFwdAlgorithm_t
get_fwd_algorithm_by_heuristics(miopenHandle_t handle,
                                miopenTensorDescriptor_t const& xdesc,
                                void const* x,
                                miopenTensorDescriptor_t const& wdesc,
                                void const* w,
                                miopenConvolutionDescriptor_t const& conv_desc,
                                miopenTensorDescriptor_t const& ydesc,
                                void* y,
                                void* ws,
                                size_t ws_size)
{
    constexpr size_t max_num_algos = 5;
    std::array<miopenConvAlgoPerf_t, max_num_algos> perf_results;
    int tested_algo_count = 0;
    DISTCONV_CHECK_MIOPEN(
        miopenFindConvolutionForwardAlgorithm(handle,
                                              xdesc,
                                              x,
                                              wdesc,
                                              w,
                                              conv_desc,
                                              ydesc,
                                              y,
                                              perf_results.size(),
                                              &tested_algo_count,
                                              perf_results.data(),
                                              ws,
                                              ws_size,
                                              /*exhaustive_search=*/0));

    for (int i = 0; i < tested_algo_count; i++)
        if (perf_results[i].memory <= ws_size)
            return perf_results[i].fwd_algo;

    util::MPIPrintStreamError() << "No forward algorithm found for MIOpen";
    std::abort();
    return miopenConvolutionFwdAlgoGEMM; // just in case a compiler whines.
}

template <typename AlgoType>
static AlgoType
find_best_algorithm(std::vector<miopenConvAlgoPerf_t> const& perf_results)
{
    std::map<AlgoType, float> time_map;
    for (auto const& res : perf_results)
    {
        auto const algo = get_algo<AlgoType>(res);
        if (time_map.find(algo) == cend(time_map))
        {
            time_map[algo] = 0;
        }
        time_map[algo] += res.time;
    }
    AlgoType best_algo = cbegin(time_map)->first;
    float min_time = std::numeric_limits<float>::max();
    for (auto const& x : time_map)
    {
        AlgoType const algo = x.first;
        float const time = x.second;
        if (time < min_time)
        {
            min_time = time;
            best_algo = algo;
        }
    }
    return best_algo;
}

// It seems there can be some variance in execution times, so multiple
// trials can be done.
static miopenConvFwdAlgorithm_t
autotune_fwd_algorithm(miopenHandle_t handle,
                       miopenTensorDescriptor_t const& xdesc,
                       void const* x,
                       miopenTensorDescriptor_t const& wdesc,
                       void const* w,
                       miopenConvolutionDescriptor_t const& conv_desc,
                       miopenTensorDescriptor_t const& ydesc,
                       void* y,
                       void* ws,
                       size_t ws_size)
{
    constexpr int trial_count = 5;
    constexpr int skip = 3;
    constexpr size_t algo_count = 5;
    std::array<miopenConvAlgoPerf_t, algo_count> perf_results;
    std::vector<miopenConvAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    for (int t = 0; t < trial_count + skip; ++t)
    {
        DISTCONV_CHECK_MIOPEN(
            miopenFindConvolutionForwardAlgorithm(handle,
                                                  xdesc,
                                                  x,
                                                  wdesc,
                                                  w,
                                                  conv_desc,
                                                  ydesc,
                                                  y,
                                                  algo_count,
                                                  &tested_algo_count,
                                                  perf_results.data(),
                                                  ws,
                                                  ws_size,
                                                  /*exhaustive_search=*/1));
        if (t > skip)
        {
            std::ostringstream oss;
            oss << "Forward autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                auto const& res = perf_results[i];
                oss << "("
                    << util::MIOpenConvolutionFwdAlgorithms::get_name(res.fwd_algo)
                    << ", "
                    << res.time << " ms"
                    << ", " << res.memory / 1000 / 1000 << " KB"
                    << ") ";
                perf_results_all.push_back(res);
            }
            util::MPIPrintStreamDebug() << oss.str();
        }
    }
    auto const best_algo =
        find_best_algorithm<miopenConvFwdAlgorithm_t>(perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::MIOpenConvolutionFwdAlgorithms::get_name(best_algo);
    return best_algo;
}

miopenConvFwdAlgorithm_t
BackendMIOpen::get_fwd_algorithm(std::string const& name,
                  miopenTensorDescriptor_t const* input_desc,
                  void const* input,
                  miopenTensorDescriptor_t const* filter_desc,
                  void const* filter,
                  miopenConvolutionDescriptor_t const* conv_desc,
                  miopenTensorDescriptor_t const* output_desc,
                  void* output,
                  size_t ws_size)
{
    std::string const n =
        (name == "DEFAULT"
             ? "HEURISTIC"
             : (name == "DETERMINISTIC" ? "IMPLICIT_GEMM" : name));

    util::MIOpenConvolutionFwdAlgorithms algos;
    for (auto const& p : algos.algo_map)
        if (p.second == n)
            return p.first;

    assert_always(input_desc);
    assert_always(filter_desc);
    assert_always(conv_desc);
    assert_always(output_desc);

    WSBuffer ws(CONVOLUTION_WORKSPACE_SIZE);
    if (n == "HEURISTIC")
        return get_fwd_algorithm_by_heuristics(get_handle(),
                                               *input_desc,
                                               input,
                                               *filter_desc,
                                               filter,
                                               *conv_desc,
                                               *output_desc,
                                               output,
                                               ws,
                                               ws_size);
    else if (n == "AUTOTUNE")
        return autotune_fwd_algorithm(get_handle(),
                                      *input_desc,
                                      input,
                                      *filter_desc,
                                      filter,
                                      *conv_desc,
                                      *output_desc,
                                      output,
                                      ws,
                                      ws_size);

    util::MPIRootPrintStreamError()
        << "No matching fwd algorithm found for MIOpen: " << n;
    std::abort();
}

static miopenConvBwdWeightsAlgorithm_t
get_bwd_data_algorithm(std::string const& name,
                       miopenHandle_t handle,
                       miopenTensorDescriptor_t const* filter_desc,
                       void const* filter,
                       miopenTensorDescriptor_t const* d_output_desc,
                       void const* d_output,
                       miopenConvolutionDescriptor_t const* conv_desc,
                       miopenTensorDescriptor_t const* d_input_desc,
                       void* d_input,
                       size_t ws_size)
{
    std::string const n =
        (name == "DEFAULT"
             ? "HEURISTIC"
             : (name == "DETERMINISTIC" ? "IMPLICIT_GEMM" : name));

    auto const& algo_map = util::MIOpenConvolutionBwdDataAlgorithms::algo_map;
    for (auto const& p : algo_map)
        if (p.second == n)
            return p.first;

    assert_always(filter_desc);
    assert_always(d_output_desc);
    assert_always(conv_desc);
    assert_always(d_input_desc);

    if (n == "HEURISTIC")
        return get_bwd_data_algorithm_by_heuristics(
            handle, *filter_desc, *d_output_desc, *conv_desc, *d_input_desc, ws_size);
    else if (n == "AUTOTUNE")
        return autotune_bwd_data_algorithm(handle,
                                           *filter_desc,
                                           filter,
                                           *d_output_desc,
                                           d_output,
                                           *conv_desc,
                                           *d_input_desc,
                                           d_input,
                                           ws_size);

    util::MPIRootPrintStreamError()
        << "No matching bwd data algorithm found for MIOpen: " << n;
    std::abort();
}

static miopenConvBwdDataAlgorithm_t get_bwd_data_algorithm_by_heuristics(
                                                                         miopenHandle_t handle,
    miopenTensorDescriptor_t const& filter_desc,
    miopenTensorDescriptor_t const& d_output_desc,
    miopenConvolutionDescriptor_t const& conv_desc,
    miopenTensorDescriptor_t const& d_input_desc,
    size_t ws_size)
{
#if 0
#if MIOpen_MAJOR < 8
    cudnnConvolutionBwdDataAlgo_t algo;
    DISTCONV_CHECK_MIOPEN(cudnnGetConvolutionBackwardDataAlgorithm(
        get_handle(),
        filter_desc,
        d_output_desc,
        conv_desc,
        d_input_desc,
        MIOpen_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE,
        &algo));
    return algo;

#else // MIOpen_MAJOR < 8
    int algo_count;
    DISTCONV_CHECK_MIOPEN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
        get_handle(), &algo_count));
    cudnnConvolutionBwdDataAlgoPerf_t* perf_results =
        new cudnnConvolutionBwdDataAlgoPerf_t[algo_count];
    int tested_algo_count = 0;
    DISTCONV_CHECK_MIOPEN(
        cudnnGetConvolutionBackwardDataAlgorithm_v7(get_handle(),
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

    util::MPIPrintStreamError() << "No backward data algorithm found for MIOpen";
    std::abort();

#endif // MIOpen_MAJOR < 8
#endif // 0
}

static miopenConvBwdDataAlgorithm_t
autotune_bwd_data_algorithm(miopenHandle_t handle,
                            miopenTensorDescriptor_t const& filter_desc,
                            void const* filter,
                            miopenTensorDescriptor_t const& d_output_desc,
                            void const* d_output,
                            miopenConvolutionDescriptor_t const& conv_desc,
                            miopenTensorDescriptor_t const& d_input_desc,
                            void* d_input,
                            size_t ws_size)
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;
    constexpr int algo_count = 4;
    std::array<miopenConvAlgoPerf_t, algo_count> perf_results;
    std::vector<miopenConvAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    void* ws = nullptr;
    if (ws_size)
    {
        ws = internal::RuntimeHIP::get_device_memory_pool().get(ws_size, 0);
    }
    for (int t = 0; t < trial_count + skip; ++t)
    {
        if (ws_size)
        {
            DISTCONV_CHECK_MIOPEN(
                miopenFindConvolutionBackwardDataAlgorithmEx(get_handle(),
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
            DISTCONV_CHECK_MIOPEN(
                miopenFindConvolutionBackwardDataAlgorithm(get_handle(),
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
            std::ostringstream oss;
            oss << "Backward data autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                auto const& res = perf_results[i];

                oss << "("
                    << util::MIOpenConvolutionBwdDataAlgorithms::get_name(
                           res.bwd_data_algo)
                    << ", "
                    << res.time << " ms"
                    << ", " << res.memory / 1000 / 1000 << " MB"
                    << ") ";
                perf_results_all.push_back(res);
            }
            util::MPIPrintStreamDebug() << ss.str();
        }
    }
    if (ws_size)
    {
        internal::RuntimeHIP::get_device_memory_pool().release(ws);
    }
    auto best_algo = find_best_algorithm<miopenConvBwdDataAlgorithm_t>(
        perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::MIOpenConvolutionBwdDataAlgorithms::get_name(best_algo);
    return best_algo;
}

miopenConvBwdWeightsAlgorithm_t BackendMIOpen::get_bwd_filter_algorithm(
    std::string const& name,
    miopenTensorDescriptor_t const* input_desc,
    void const* input,
    miopenTensorDescriptor_t const* d_output_desc,
    void const* d_output,
    miopenConvolutionDescriptor_t const* conv_desc,
    miopenTensorDescriptor_t const* d_filter_desc,
    void* d_filter,
    size_t ws_size)
{
    std::string const n =
        (name == "DEFAULT"
             ? "HEURISTIC"
             : (name == "DETERMINISTIC" ? "IMPLICIT_GEMM" : name));

    auto const& algo_map = util::MIOpenConvolutionBwdWeightsAlgorithms::algo_map;
    for (auto const& p : algo_map)
    {
        if (p.second == n)
            return p.first;
    }

    assert_always(input_desc);
    assert_always(d_output_desc);
    assert_always(conv_desc);
    assert_always(d_filter_desc);

    if (n == "HEURISTIC")
    {
        return get_bwd_filter_algorithm_by_heuristics(
            *input_desc, *d_output_desc, *conv_desc, *d_filter_desc, ws_size);
    }
    else if (n == "AUTOTUNE")
    {
        return autotune_bwd_filter_algorithm(*input_desc,
                                             input,
                                             *d_output_desc,
                                             d_output,
                                             *conv_desc,
                                             *d_filter_desc,
                                             d_filter,
                                             ws_size);
    }

    util::MPIRootPrintStreamError()
        << "No matching bwd filter algorithm found for MIOpen: " << n;
    std::abort();
}

miopenConvBwdWeightsAlgorithm_t
BackendMIOpen::get_bwd_filter_algorithm_by_heuristics(
    miopenTensorDescriptor_t const& input_desc,
    miopenTensorDescriptor_t const& d_output_desc,
    miopenConvolutionDescriptor_t const& conv_desc,
    miopenTensorDescriptor_t const& d_filter_desc,
    size_t ws_size)
{
#if 0
#if MIOpen_MAJOR < 8
    miopenConvBwdWeightsAlgorithm_t algo;
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionBackwardFilterAlgorithm(
        get_handle(),
        input_desc,
        d_output_desc,
        conv_desc,
        d_filter_desc,
        MIOpen_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
        ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE,
        &algo));
    return algo;

#else // MIOpen_MAJOR < 8
    int algo_count;
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionBackwardFilterAlgorithmMaxCount(
        get_handle(), &algo_count));
    miopenConvolutionBwdWeightsAlgoPerf_t* perf_results =
        new miopenConvolutionBwdWeightsAlgoPerf_t[algo_count];
    int tested_algo_count = 0;
    DISTCONV_CHECK_MIOPEN(
        miopenGetConvolutionBackwardFilterAlgorithm_v7(get_handle(),
                                                       input_desc,
                                                       d_output_desc,
                                                       conv_desc,
                                                       d_filter_desc,
                                                       algo_count,
                                                       &tested_algo_count,
                                                       perf_results));

    miopenConvBwdWeightsAlgorithm_t algo;
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
        << "No backward filter algorithm found for MIOpen";
    std::abort();

#endif // MIOpen_MAJOR < 8
#endif // 0
}

miopenConvBwdWeightsAlgorithm_t BackendMIOpen::autotune_bwd_filter_algorithm(
    miopenTensorDescriptor_t const& input_desc,
    void const* input,
    miopenTensorDescriptor_t const& d_output_desc,
    void const* d_output,
    miopenConvolutionDescriptor_t const& conv_desc,
    miopenTensorDescriptor_t const& d_filter_desc,
    void* d_filter,
    size_t ws_size)
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;

    std::array<miopenConvAlgoPerf_t, 5> perf_results;
    std::vector<miopenConvAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    void* ws = nullptr;
    if (ws_size)
    {
        ws = internal::RuntimeHIP::get_device_memory_pool().get(ws_size, 0);
    }
    for (int t = 0; t < trial_count + skip; ++t)
    {
        if (ws_size)
        {
            DISTCONV_CHECK_MIOPEN(
                miopenFindConvolutionBackwardFilterAlgorithmEx(
                    get_handle(),
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
            DISTCONV_CHECK_MIOPEN(
                miopenFindConvolutionBackwardFilterAlgorithm(get_handle(),
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
            std::ostringstream oss;
            oss << "Backward filter autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                auto const& res = perf_results[i];

                oss << "("
                    << util::MIOpenConvolutionBwdWeightsAlgorithms::get_name(
                           res.bwd_weights_algo)
                    << ", "
                    << res.time << " ms"
                    << ", " << res.memory / 1000 / 1000 << " MB"
                    << ") ";
                perf_results_all.push_back(res);
            }
            util::MPIPrintStreamDebug() << ss.str();
        }
    }
    if (ws_size)
    {
        internal::RuntimeHIP::get_device_memory_pool().release(ws);
    }
    auto const best_algo =
        find_best_algorithm<miopenConvBwdWeightsAlgorithm_t>(perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::MIOpenConvolutionBwdWeightsAlgorithms::get_name(best_algo);
    return best_algo;
}

} // namespace miopen
} // namespace distconv
