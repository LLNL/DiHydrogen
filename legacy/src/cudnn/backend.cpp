#include "distconv/cudnn/backend.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cudnn.hpp"

#include <limits>

namespace distconv {
namespace cudnn {

// Default workspace wize
constexpr size_t CONVOLUTION_WORKSPACE_SIZE = 1 << 30;

cudnnConvolutionFwdAlgo_t BackendCUDNN::get_fwd_algorithm(
    const std::string &name,
    const cudnnTensorDescriptor_t *input_desc,
    const void *input,
    const cudnnFilterDescriptor_t *filter_desc,
    const void *filter,
    const cudnnConvolutionDescriptor_t *conv_desc,
    const cudnnTensorDescriptor_t *output_desc,
    void *output,
    size_t ws_size) {
  std::string n = name;
  if (name == "DEFAULT") {
    // Default selection
    n = "HEURISTIC";
  } else if (name == "DETERMINISTIC") {
    // Use IMPLICIT_GEMM as it is deterministic
    n = "IMPLICIT_GEMM";
  }

  util::CUDNNConvolutionFwdAlgorithms algos;
  for (const auto &p: algos.algos) {
    if (p.second == n) return p.first;
  }

  assert_always(input_desc);
  assert_always(filter_desc);
  assert_always(conv_desc);
  assert_always(output_desc);

  if (n == "HEURISTIC") {
    return get_fwd_algorithm_by_heuristics(
        *input_desc, *filter_desc, *conv_desc, *output_desc,
        ws_size);
  } else if (n == "AUTOTUNE") {
    return autotune_fwd_algorithm(
        *input_desc, input, *filter_desc, filter, *conv_desc, *output_desc,
        output, ws_size);
  }

  util::MPIRootPrintStreamError()
      << "No matching fwd algorithm found for CUDNN: " << n;
  std::abort();
}

cudnnConvolutionFwdAlgo_t BackendCUDNN::get_fwd_algorithm_by_heuristics(
    const cudnnTensorDescriptor_t &input_desc,
    const cudnnFilterDescriptor_t &filter_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &output_desc,
    size_t ws_size) {
#if CUDNN_MAJOR < 8
  cudnnConvolutionFwdAlgo_t algo;
  DISTCONV_CHECK_CUDNN(
      cudnnGetConvolutionForwardAlgorithm(
          get_handle(), input_desc, filter_desc, conv_desc,
          output_desc, CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
          ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE, &algo));
  return algo;

#else // CUDNN_MAJOR < 8
  util::MPIPrintStreamError() << "cudnnGetConvolutionForwardAlgorithm is deprecated."
                              << " Use cudnnFindConvolutionForwardAlgorithm instead.";
  throw std::exception();

#endif // CUDNN_MAJOR < 8
}

template <typename AlgoType, typename PerfType>
AlgoType find_best_algorithm(const std::vector<PerfType> &perf_results) {
  std::map<AlgoType, float> time_map;
  for (const auto &res: perf_results) {
    assert_always(res.status == CUDNN_STATUS_SUCCESS);
    if (time_map.find(res.algo) == time_map.end()) {
      time_map[res.algo] = 0;
    }
    time_map[res.algo] += res.time;
  }
  AlgoType best_algo = time_map.begin()->first;
  float min_time = std::numeric_limits<float>::max();
  for (const auto &x: time_map) {
    AlgoType algo = x.first;
    float time = x.second;
    if (time < min_time) {
      min_time = time;
      best_algo = algo;
    }
  }
  return best_algo;
}

// It seems there can be some variance in execution times, so multiple
// trials can be done.
cudnnConvolutionFwdAlgo_t BackendCUDNN::autotune_fwd_algorithm(
    const cudnnTensorDescriptor_t &input_desc,
    const void *input,
    const cudnnFilterDescriptor_t &filter_desc,
    const void *filter,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &output_desc,
    void *output,
    size_t ws_size) {
  constexpr int trial_count = 5;
  constexpr int skip = 5;
  int algo_count;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(
      get_handle(), &algo_count));
  cudnnConvolutionFwdAlgoPerf_t *perf_results = new
      cudnnConvolutionFwdAlgoPerf_t[algo_count];
  std::vector<cudnnConvolutionFwdAlgoPerf_t> perf_results_all;
  int tested_algo_count = 0;
  void *ws = nullptr;
  if (ws_size) {
    ws = internal::RuntimeCUDA::get_device_memory_pool().get(
        ws_size, 0);
  }
  for (int t = 0; t < trial_count + skip; ++t) {
    if (ws_size) {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithmEx(
          get_handle(), input_desc, input, filter_desc, filter,
          conv_desc, output_desc, output, algo_count, &tested_algo_count,
          perf_results, ws, ws_size));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(
          get_handle(), input_desc, filter_desc, conv_desc,
          output_desc, algo_count, &tested_algo_count,
          perf_results));
    }
    if (t > skip) {
      std::stringstream ss;
      ss << "Forward autotune tested algorithms: ";
      for (int i = 0; i < tested_algo_count; ++i) {
        const auto &res = perf_results[i];
        ss << "("
           << util::CUDNNConvolutionFwdAlgorithms::get_name(res.algo)
           << ", ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          ss << res.time << " ms"
             << ", " << res.memory / 1000 / 1000 << " KB";
        } else if (res.status == CUDNN_STATUS_ALLOC_FAILED) {
          ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
             << " MB required";
        } else {
          ss << "INTERNAL ERROR";
        }
        ss << ") ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          perf_results_all.push_back(res);
        }
      }
      util::MPIPrintStreamDebug() << ss.str();
    }
  }
  delete[] perf_results;
  if (ws_size) {
    internal::RuntimeCUDA::get_device_memory_pool().release(ws);
  }
  auto best_algo = find_best_algorithm<
    cudnnConvolutionFwdAlgo_t, cudnnConvolutionFwdAlgoPerf_t>(
        perf_results_all);
  util::MPIPrintStreamDebug()
      << "Autotune best algorithm: "
      << util::CUDNNConvolutionFwdAlgorithms::get_name(best_algo);
  return best_algo;
}

cudnnConvolutionBwdDataAlgo_t BackendCUDNN::get_bwd_data_algorithm(
    const std::string &name,
    const cudnnFilterDescriptor_t *filter_desc,
    const void *filter,
    const cudnnTensorDescriptor_t *d_output_desc,
    const void *d_output,
    const cudnnConvolutionDescriptor_t *conv_desc,
    const cudnnTensorDescriptor_t *d_input_desc,
    void *d_input,
    size_t ws_size) {
  std::string n = name;
  if (name == "DEFAULT") {
    // Default selection
    n = "HEURISTIC";
  } else if (name == "DETERMINISTIC") {
    // Use ALGO_1 as it is deterministic
    n = "ALGO_1";
  }

  util::CUDNNConvolutionBwdDataAlgorithms algos;
  for (const auto &p: algos.algos) {
    if (p.second == n) return p.first;
  }

  assert_always(filter_desc);
  assert_always(d_output_desc);
  assert_always(conv_desc);
  assert_always(d_input_desc);

  if (n == "HEURISTIC") {
    return get_bwd_data_algorithm_by_heuristics(
        *filter_desc, *d_output_desc, *conv_desc, *d_input_desc,
        ws_size);
  } else if (n == "AUTOTUNE") {
    return autotune_bwd_data_algorithm(
        *filter_desc, filter, *d_output_desc, d_output, *conv_desc,
        *d_input_desc, d_input, ws_size);
  }

  util::MPIRootPrintStreamError()
      << "No matching bwd data algorithm found for CUDNN: " << n;
  std::abort();
}

cudnnConvolutionBwdDataAlgo_t BackendCUDNN::get_bwd_data_algorithm_by_heuristics(
    const cudnnFilterDescriptor_t &filter_desc,
    const cudnnTensorDescriptor_t &d_output_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &d_input_desc,
    size_t ws_size) {
#if CUDNN_MAJOR < 8
  cudnnConvolutionBwdDataAlgo_t algo;
  DISTCONV_CHECK_CUDNN(
      cudnnGetConvolutionBackwardDataAlgorithm(
          get_handle(), filter_desc, d_output_desc, conv_desc,
          d_input_desc, CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
          ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE, &algo));
  return algo;

#else // CUDNN_MAJOR < 8
  util::MPIPrintStreamError() << "cudnnGetConvolutionBackwardDataAlgorithm is deprecated."
                              << " Use cudnnFindConvolutionBackwardDataAlgorithm instead.";
  throw std::exception();

#endif // CUDNN_MAJOR < 8
}

cudnnConvolutionBwdDataAlgo_t BackendCUDNN::autotune_bwd_data_algorithm(
    const cudnnFilterDescriptor_t &filter_desc,
    const void *filter,
    const cudnnTensorDescriptor_t &d_output_desc,
    const void *d_output,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnTensorDescriptor_t &d_input_desc,
    void *d_input,
    size_t ws_size) {
  constexpr int trial_count = 3;
  constexpr int skip = 1;
  int algo_count;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
      get_handle(), &algo_count));
  cudnnConvolutionBwdDataAlgoPerf_t *perf_results = new
      cudnnConvolutionBwdDataAlgoPerf_t[algo_count];
  std::vector<cudnnConvolutionBwdDataAlgoPerf_t> perf_results_all;
  int tested_algo_count = 0;
  void *ws = nullptr;
  if (ws_size) {
    ws = internal::RuntimeCUDA::get_device_memory_pool().get(
        ws_size, 0);
  }
  for (int t = 0; t < trial_count + skip; ++t) {
    if (ws_size) {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithmEx(
          get_handle(), filter_desc, filter, d_output_desc, d_output,
          conv_desc, d_input_desc, d_input, algo_count,
          &tested_algo_count, perf_results, ws, ws_size));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionBackwardDataAlgorithm(
          get_handle(), filter_desc, d_output_desc, conv_desc,
          d_input_desc, algo_count, &tested_algo_count,
          perf_results));
    }
    if (t > skip) {
      std::stringstream ss;
      ss << "Backward data autotune tested algorithms: ";
      for (int i = 0; i < tested_algo_count; ++i) {
        const auto &res = perf_results[i];

        ss << "("
           << util::CUDNNConvolutionBwdDataAlgorithms::get_name(res.algo)
           << ", ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          ss << res.time << " ms"
             << ", " << res.memory / 1000 / 1000 << " MB";
        } else if (res.status == CUDNN_STATUS_ALLOC_FAILED) {
          ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
             << " MB required";
        } else {
          ss << "INTERNAL ERROR";
        }
        ss << ") ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          perf_results_all.push_back(res);
        }
      }
      util::MPIPrintStreamDebug() << ss.str();
    }
  }
  delete[] perf_results;
  if (ws_size) {
    internal::RuntimeCUDA::get_device_memory_pool().release(ws);
  }
  auto best_algo = find_best_algorithm<
    cudnnConvolutionBwdDataAlgo_t, cudnnConvolutionBwdDataAlgoPerf_t>(
        perf_results_all);
  util::MPIPrintStreamDebug()
      << "Autotune best algorithm: "
      << util::CUDNNConvolutionBwdDataAlgorithms::get_name(best_algo);
  return best_algo;
}

cudnnConvolutionBwdFilterAlgo_t BackendCUDNN::get_bwd_filter_algorithm(
    const std::string &name,
    const cudnnTensorDescriptor_t *input_desc,
    const void *input,
    const cudnnTensorDescriptor_t *d_output_desc,
    const void *d_output,
    const cudnnConvolutionDescriptor_t *conv_desc,
    const cudnnFilterDescriptor_t *d_filter_desc,
    void *d_filter,
    size_t ws_size) {
  std::string n = name;
  if (name == "DEFAULT") {
    // Default selection
    n = "HEURISTIC";
  } else if (name == "DETERMINISTIC") {
    // Use ALGO_1 as it is deterministic
    n = "ALGO_1";
  }

  util::CUDNNConvolutionBwdFilterAlgorithms algos;
  for (const auto &p: algos.algos) {
    if (p.second == n) return p.first;
  }

  assert_always(input_desc);
  assert_always(d_output_desc);
  assert_always(conv_desc);
  assert_always(d_filter_desc);

  if (n == "HEURISTIC") {
    return get_bwd_filter_algorithm_by_heuristics(
        *input_desc, *d_output_desc, *conv_desc, *d_filter_desc,
        ws_size);
  } else if (n == "AUTOTUNE") {
    return autotune_bwd_filter_algorithm(
        *input_desc, input, *d_output_desc, d_output, *conv_desc,
        *d_filter_desc, d_filter, ws_size);
  }

  util::MPIRootPrintStreamError()
      << "No matching bwd filter algorithm found for CUDNN: " << n;
  std::abort();
}

cudnnConvolutionBwdFilterAlgo_t
BackendCUDNN::get_bwd_filter_algorithm_by_heuristics(
    const cudnnTensorDescriptor_t &input_desc,
    const cudnnTensorDescriptor_t &d_output_desc,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnFilterDescriptor_t &d_filter_desc,
    size_t ws_size) {
#if CUDNN_MAJOR < 8
  cudnnConvolutionBwdFilterAlgo_t algo;
  DISTCONV_CHECK_CUDNN(
      cudnnGetConvolutionBackwardFilterAlgorithm(
          get_handle(), input_desc, d_output_desc, conv_desc,
          d_filter_desc,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          ws_size ? ws_size : CONVOLUTION_WORKSPACE_SIZE, &algo));
  return algo;

#else // CUDNN_MAJOR < 8
  util::MPIPrintStreamError() << "cudnnGetConvolutionBackwardFilterAlgorithm is deprecated."
                              << " Use cudnnFindConvolutionBackwardFilterAlgorithm instead.";
  throw std::exception();

#endif // CUDNN_MAJOR < 8
}

cudnnConvolutionBwdFilterAlgo_t BackendCUDNN::autotune_bwd_filter_algorithm(
    const cudnnTensorDescriptor_t &input_desc,
    const void *input,
    const cudnnTensorDescriptor_t &d_output_desc,
    const void *d_output,
    const cudnnConvolutionDescriptor_t &conv_desc,
    const cudnnFilterDescriptor_t &d_filter_desc,
    void *d_filter,
    size_t ws_size) {
  constexpr int trial_count = 3;
  constexpr int skip = 1;
  int algo_count;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
      get_handle(), &algo_count));
  cudnnConvolutionBwdFilterAlgoPerf_t *perf_results = new
      cudnnConvolutionBwdFilterAlgoPerf_t[algo_count];
  std::vector<cudnnConvolutionBwdFilterAlgoPerf_t> perf_results_all;
  int tested_algo_count = 0;
  void *ws = nullptr;
  if (ws_size) {
    ws = internal::RuntimeCUDA::get_device_memory_pool().get(
        ws_size, 0);
  }
  for (int t = 0; t < trial_count + skip; ++t) {
    if (ws_size) {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithmEx(
          get_handle(), input_desc, input, d_output_desc, d_output,
          conv_desc, d_filter_desc, d_filter, algo_count, &tested_algo_count,
          perf_results, ws, ws_size));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnFindConvolutionBackwardFilterAlgorithm(
          get_handle(), input_desc, d_output_desc, conv_desc,
          d_filter_desc, algo_count, &tested_algo_count,
          perf_results));
    }
    if (t > skip) {
      std::stringstream ss;
      ss << "Backward filter autotune tested algorithms: ";
      for (int i = 0; i < tested_algo_count; ++i) {
        const auto &res = perf_results[i];

        ss << "("
           << util::CUDNNConvolutionBwdFilterAlgorithms::get_name(res.algo)
           << ", ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          ss << res.time << " ms"
             << ", " << res.memory / 1000 / 1000 << " MB";
        } else if (res.status == CUDNN_STATUS_ALLOC_FAILED) {
          ss << "INSUFFICIENT MEMORY, " << res.memory / 1000 / 1000
             << " MB required";
        } else {
          ss << "INTERNAL ERROR";
        }
        ss << ") ";
        if (res.status == CUDNN_STATUS_SUCCESS) {
          perf_results_all.push_back(res);
        }
      }
      util::MPIPrintStreamDebug() << ss.str();
    }
  }
  delete[] perf_results;
  if (ws_size) {
    internal::RuntimeCUDA::get_device_memory_pool().release(ws);
  }
  auto best_algo = find_best_algorithm<
    cudnnConvolutionBwdFilterAlgo_t, cudnnConvolutionBwdFilterAlgoPerf_t>(
        perf_results_all);
  util::MPIPrintStreamDebug()
      << "Autotune best algorithm: "
      << util::CUDNNConvolutionBwdFilterAlgorithms::get_name(best_algo);
  return best_algo;
}

} // namespace cudnn
} // namespace distconv
