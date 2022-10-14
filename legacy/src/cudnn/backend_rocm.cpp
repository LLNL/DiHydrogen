#include "distconv/cudnn/backend_miopen.hpp"
#include "distconv/util/util_miopen.hpp"
#include "distconv/util/util_mpi.hpp"

#include <limits>
#include <string>

namespace
{
struct WSBuffer
{
    WSBuffer(size_t bytes)
        : m_buffer{distconv::internal::RuntimeHIP::get_device_memory_pool().get(bytes, 0)}
    {
        assert_always((bool) m_buffer);
    }
    ~WSBuffer()
    {
        if (m_buffer)
            distconv::internal::RuntimeHIP::get_device_memory_pool().release(m_buffer);
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

static std::unordered_map<miopenPoolingDescriptor_t, void*> workspace_map;
}// namespace

void distconv::miopen::details::set_workspace(miopenPoolingDescriptor_t const& desc, void* workspace)
{
    workspace_map[desc] = workspace;
}
void* distconv::miopen::details::get_workspace(miopenPoolingDescriptor_t const& desc)
{
    return workspace_map.at(desc);
}
void distconv::miopen::details::clear_workspace(miopenPoolingDescriptor_t const& desc)
{
    if (workspace_map.count(desc))
    {
        ::distconv::internal::RuntimeHIP::get_device_memory_pool().release(
            workspace_map[desc]);
        workspace_map.erase(desc);
    }
}
std::pair<void*, size_t>
distconv::miopen::details::make_workspace(miopenHandle_t handle,
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

namespace distconv
{
namespace miopen
{

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
                                              CONVOLUTION_WORKSPACE_SIZE,
                                              /*exhaustive_search=*/1));
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
                                                  CONVOLUTION_WORKSPACE_SIZE,
                                                  /*exhaustive_search=*/1));
        if (t > skip)
        {
            std::ostringstream oss;
            oss << "Forward autotune tested algorithms: ";
            for (int i = 0; i < tested_algo_count; ++i)
            {
                auto const& res = perf_results[i];
                oss << "("
                    << util::MIOpenConvolutionFwdAlgorithms::get_name(
                           res.fwd_algo)
                    << ", " << res.time << " ms"
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

// FIXME (trb 08/11/2022): CLEANUP THE ws_size ARGUMENT TO THE MIOpen CALLS!
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

static miopenConvBwdDataAlgorithm_t
get_bwd_data_algorithm_by_heuristics(miopenHandle_t handle,
                                     miopenTensorDescriptor_t const& filter_desc,
                                     void const* filter,
                                     miopenTensorDescriptor_t const& d_output_desc,
                                     void const* d_output,
                                     miopenConvolutionDescriptor_t const& conv_desc,
                                     miopenTensorDescriptor_t const& d_input_desc,
                                     void* d_input,
                                     void* ws,
                                     size_t ws_size)
{
    constexpr size_t max_algo_count = 5;
    std::array<miopenConvAlgoPerf_t, max_algo_count> perf_results;
    int tested_algo_count = -1;
    DISTCONV_CHECK_MIOPEN(
        miopenFindConvolutionBackwardDataAlgorithm(
            handle,
            d_output_desc,
            d_output,
            filter_desc,
            filter,
            conv_desc,
            d_input_desc,
            d_input,
            max_algo_count,
            &tested_algo_count,
            perf_results.data(),
            ws,
            CONVOLUTION_WORKSPACE_SIZE,
            /*exhaustiveSearch=*/1));

    for (int i = 0; i < tested_algo_count; i++)
        if (perf_results[i].memory <= ws_size)
            return perf_results[i].bwd_data_algo;

    util::MPIPrintStreamError() << "No backward data algorithm found for MIOpen";
    std::abort();
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
                            void* ws,
                            size_t ws_size)
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;
    constexpr int max_algo_count = 5;
    std::array<miopenConvAlgoPerf_t, max_algo_count> perf_results;
    std::vector<miopenConvAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    for (int t = 0; t < trial_count + skip; ++t)
    {
        DISTCONV_CHECK_MIOPEN(
            miopenFindConvolutionBackwardDataAlgorithm(handle,
                                                       d_output_desc,
                                                       d_output,
                                                       filter_desc,
                                                       filter,
                                                       conv_desc,
                                                       d_input_desc,
                                                       d_input,
                                                       max_algo_count,
                                                       &tested_algo_count,
                                                       perf_results.data(),
                                                       ws,
                                                       CONVOLUTION_WORKSPACE_SIZE,
                                                       /*exhaustiveSearch=*/1));

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
                    << ", " << res.time << " ms"
                    << ", " << res.memory / 1000 / 1000 << " MB"
                    << ") ";
                perf_results_all.push_back(res);
            }
            util::MPIPrintStreamDebug() << oss.str();
        }
    }
    auto const best_algo =
        find_best_algorithm<miopenConvBwdDataAlgorithm_t>(perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::MIOpenConvolutionBwdDataAlgorithms::get_name(best_algo);
    return best_algo;
}

miopenConvBwdDataAlgorithm_t
BackendMIOpen::get_bwd_data_algorithm(std::string const& name,
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

    WSBuffer ws(CONVOLUTION_WORKSPACE_SIZE);
    if (n == "HEURISTIC")
        return get_bwd_data_algorithm_by_heuristics(get_handle(),
                                                    *filter_desc,
                                                    filter,
                                                    *d_output_desc,
                                                    d_output,
                                                    *conv_desc,
                                                    *d_input_desc,
                                                    d_input,
                                                    ws,
                                                    ws_size);
    else if (n == "AUTOTUNE")
        return autotune_bwd_data_algorithm(get_handle(),
                                           *filter_desc,
                                           filter,
                                           *d_output_desc,
                                           d_output,
                                           *conv_desc,
                                           *d_input_desc,
                                           d_input,
                                           ws,
                                           ws_size);

    util::MPIRootPrintStreamError()
        << "No matching bwd data algorithm found for MIOpen: " << n;
    std::abort();
}

static miopenConvBwdWeightsAlgorithm_t
get_bwd_weights_algorithm_by_heuristics(miopenHandle_t handle,
                                        miopenTensorDescriptor_t const& input_desc,
                                        void const* input,
                                        miopenTensorDescriptor_t const& d_output_desc,
                                        void const* d_output,
                                        miopenConvolutionDescriptor_t const& conv_desc,
                                        miopenTensorDescriptor_t const& d_filter_desc,
                                        void* d_filter,
                                        void* ws,
                                        size_t ws_size)
{
    constexpr int max_algo_count = 5;
    std::array<miopenConvAlgoPerf_t, max_algo_count> perf_results;
    int tested_algo_count = -1;
    DISTCONV_CHECK_MIOPEN(
        miopenFindConvolutionBackwardWeightsAlgorithm(
            handle,
            d_output_desc,
            d_output,
            input_desc,
            input,
            conv_desc,
            d_filter_desc,
            d_filter,
            max_algo_count,
            &tested_algo_count,
            perf_results.data(),
            ws,
            CONVOLUTION_WORKSPACE_SIZE,
            /*exhaustiveSearch=*/1));

    for (int i = 0; i < tested_algo_count; i++)
        if (perf_results[i].memory <= ws_size)
            return perf_results[i].bwd_weights_algo;

    util::MPIPrintStreamError()
        << "No backward filter algorithm found for MIOpen";
    std::abort();
}

static miopenConvBwdWeightsAlgorithm_t
autotune_bwd_weights_algorithm(miopenHandle_t handle,
                               miopenTensorDescriptor_t const& input_desc,
                               void const* input,
                               miopenTensorDescriptor_t const& d_output_desc,
                               void const* d_output,
                               miopenConvolutionDescriptor_t const& conv_desc,
                               miopenTensorDescriptor_t const& d_filter_desc,
                               void* d_filter,
                               void* ws,
                               size_t ws_size)
{
    constexpr int trial_count = 3;
    constexpr int skip = 1;
    constexpr int max_algo_count = 5;
    std::array<miopenConvAlgoPerf_t, max_algo_count> perf_results;
    std::vector<miopenConvAlgoPerf_t> perf_results_all;
    int tested_algo_count = 0;
    for (int t = 0; t < trial_count + skip; ++t)
    {
        DISTCONV_CHECK_MIOPEN(
            miopenFindConvolutionBackwardWeightsAlgorithm(
                handle,
                d_output_desc,
                d_output,
                input_desc,
                input,
                conv_desc,
                d_filter_desc,
                d_filter,
                max_algo_count,
                &tested_algo_count,
                perf_results.data(),
                ws,
                CONVOLUTION_WORKSPACE_SIZE,
                /*exhaustiveSearch=*/1));
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
                    << ", " << res.time << " ms"
                    << ", " << res.memory / 1000 / 1000 << " MB"
                    << ") ";
                perf_results_all.push_back(res);
            }
            util::MPIPrintStreamDebug() << oss.str();
        }
    }
    auto const best_algo =
        find_best_algorithm<miopenConvBwdWeightsAlgorithm_t>(perf_results_all);
    util::MPIPrintStreamDebug()
        << "Autotune best algorithm: "
        << util::MIOpenConvolutionBwdWeightsAlgorithms::get_name(best_algo);
    return best_algo;
}

miopenConvBwdWeightsAlgorithm_t
BackendMIOpen::get_bwd_filter_algorithm(
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

    auto const& algo_map =
        util::MIOpenConvolutionBwdWeightsAlgorithms::algo_map;
    for (auto const& p : algo_map)
    {
        if (p.second == n)
            return p.first;
    }

    assert_always(input_desc);
    assert_always(d_output_desc);
    assert_always(conv_desc);
    assert_always(d_filter_desc);

    WSBuffer ws(CONVOLUTION_WORKSPACE_SIZE);
    if (n == "HEURISTIC")
    {
        return get_bwd_weights_algorithm_by_heuristics(get_handle(),
                                                       *input_desc,
                                                       input,
                                                       *d_output_desc,
                                                       d_output,
                                                       *conv_desc,
                                                       *d_filter_desc,
                                                       d_filter,
                                                       ws,
                                                       ws_size);
    }
    else if (n == "AUTOTUNE")
    {
        return autotune_bwd_weights_algorithm(get_handle(),
                                              *input_desc,
                                              input,
                                              *d_output_desc,
                                              d_output,
                                              *conv_desc,
                                              *d_filter_desc,
                                              d_filter,
                                              ws,
                                              ws_size);
    }

    util::MPIRootPrintStreamError()
        << "No matching bwd filter algorithm found for MIOpen: " << n;
    std::abort();
}

} // namespace miopen
} // namespace distconv
