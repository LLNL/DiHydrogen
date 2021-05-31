#pragma once

#include "distconv/base.hpp"
#include "distconv/vector.hpp"
#include "distconv/util/cxxopts.hpp"
#include "distconv/tensor/tensor_base.hpp"
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <numeric>

namespace distconv_benchmark {

using distconv::Vector;
using distconv::IndexVector;
using distconv::tensor::Shape;
using distconv::int_vector;

enum class BenchmarkDataType {FLOAT, DOUBLE, HALF};

template <BenchmarkDataType type>
struct GetType;

template <>
struct GetType<BenchmarkDataType::FLOAT> {
  using type = float;
};

template <>
struct GetType<BenchmarkDataType::DOUBLE> {
  using type = double;
};

template <>
struct GetType<BenchmarkDataType::HALF> {
  using type = half;
};

const unsigned input_tensor_seed = 0;
const unsigned filter_tensor_seed = 1;
const unsigned d_output_tensor_seed = 2;

template <typename REAL>
struct Initializer {
  Initializer(unsigned seed) {
    std::srand(seed);
    for (int i = 0; i < num_rands; ++i) {
      m_rands[i] = std::rand();
    }
  }

  REAL get_initial_value(const IndexVector &indices,
                         const Shape &dims) {
    const auto x = indices.reduce_sum();
    const double rand1 = (double)(m_rands[x % num_rands]) / RAND_MAX;
    size_t offset = 0;
    { // offset = w_idx + h_idx * w_dim + c_idx * w_dim * h_dim + ...;
      size_t d_prod = 1;
      for(auto i = indices.begin(); i != indices.end(); i++) {
        offset += *i * d_prod;
        d_prod *= *(dims.begin()+std::distance(indices.begin(), i));
      }
    }
    const double rand2 = (double)(m_rands[offset % num_rands]) / RAND_MAX;
    const REAL v = (rand1 + rand2) / 2;
    return v;
  }

  REAL get_initial_value(const std::vector<size_t> indices,
                         const std::vector<size_t> dims) {
    return get_initial_value(IndexVector(indices), Shape(dims));
  }

  static constexpr int num_rands = 100;
  unsigned m_rands[num_rands];
};

template <typename T>
inline T get_median(const std::vector<T> &v) {
  std::vector<T> tmp = v;
  int mean_idx = tmp.size() / 2 - 1 + tmp.size() % 2;
  std::nth_element(tmp.begin(), tmp.begin() + mean_idx, tmp.end());
  return tmp[mean_idx];
}

template <typename T>
inline T get_mean(const std::vector<T> &v) {
  return std::accumulate(v.begin(), v.end(), 0.0) /
      v.size();
}

template <typename T>
inline T get_min(const std::vector<T> &v) {
  return *std::min_element(v.begin(), v.end());
}

template <typename T>
inline T get_max(const std::vector<T> &v) {
  return *std::max_element(v.begin(), v.end());
}

inline int get_dilated_filter_size(int filter_size, int dilation) {
  return filter_size + (filter_size - 1) * (dilation - 1);
}

template <int NSD>
class BenchmarkConfig {
 public:
  // Input tensor dimensions
  int i_n;
  int i_c;
  int_vector i_s;

  int f_k;
  int_vector f_s;

  int_vector pads;

  bool use_padding;
  bool use_bias;

  int_vector strides;

  int_vector dilations;

  int num_groups;

  // Dimensions of process grid
  int p_n;
  int p_c;
  int_vector p_s;
  int p_f;

  int warming_up_count;
  int run_count;

  std::string conv_fwd_algo;
  std::string conv_bwd_data_algo;
  std::string conv_bwd_filter_algo;

  std::string pooling_mode;

  std::string backend;

  std::string output_file;
  bool dump_input;
  bool dump_output;
  bool dump_binary;

  BenchmarkDataType data_type;

  enum mode_t {NORMAL, SIMPLE};
  mode_t mode;

  bool profiling;
  bool nvtx_marking;

  distconv::HaloExchangeMethod halo_exchange_method;
  bool overlap_halo_exchange;

  distconv::ShuffleMethod shuffle_method;

  distconv::BatchnormImpl batchnorm_impl;

  bool deterministic;

  bool skip_weight_allreduce;
  bool skip_halo_exchange;
  bool testing;

  distconv::ChannelParallelismAlgorithm chanfilt_algo;
  bool skip_chanfilt_comm;

  int spin_time_ms;

  bool host;

  bool global_stat;

  bool deconv;

  // Some initial values are intended to be rewritten by corresponding
  // default/user-given arguments in `cxxopts::ParseResult`.
  BenchmarkConfig(): i_n(-1), i_c(-1), i_s({}),
            f_k(-1), f_s({}),
            pads({}),
            use_padding(true),
            use_bias(false),
            strides({}),
            dilations({}),
            num_groups(1),
            p_n(-1), p_c(-1), p_s({}), p_f(-1),
            warming_up_count(-1), run_count(-1),
            conv_fwd_algo("DEFAULT"),
            conv_bwd_data_algo("DEFAULT"),
            conv_bwd_filter_algo("DEFAULT"),
            pooling_mode("MAX"),
            backend("CUDNN"),
            output_file("results"),
            dump_input(false),
            dump_output(false),
            dump_binary(false),
            data_type(BenchmarkDataType::FLOAT),
            mode(mode_t::NORMAL),
            profiling(false),
            nvtx_marking(false),
            overlap_halo_exchange(false),
            deterministic(false),
            skip_weight_allreduce(false),
            skip_halo_exchange(false),
            testing(false),
            chanfilt_algo(distconv::ChannelParallelismAlgorithm::NONE),
            skip_chanfilt_comm(false),
            spin_time_ms(100),
            host(false),
            global_stat(false),
            deconv(false) {}
  BenchmarkConfig(const cxxopts::ParseResult &pr, const bool is_conv):
      BenchmarkConfig() {
    // The following arguments are required.
    run_count = pr["num-runs"].as<int>();
    warming_up_count = pr["num-warmup-runs"].as<int>();
    output_file = pr["output-file"].as<std::string>();
    substitute_nd_argument(i_n, i_c, i_s, pr["image-size"].as<std::string>());
    {
      const auto s = pr["filter-size"].as<std::string>();
      if(is_conv)
        substitute_nd_argument(f_k, f_s, s);
      else
        substitute_nd_argument(f_s, s);
    }
    use_padding = !pr["no-padding"].as<bool>();
    use_bias = pr["use-bias"].as<bool>();
    strides = distconv::util::reverse(
        distconv::util::split_spaced_array<int>(pr["strides"].as<std::string>()));
    dilations = distconv::util::reverse(
        distconv::util::split_spaced_array<int>(pr["dilations"].as<std::string>()));
    num_groups = pr["num-groups"].as<int>();
    substitute_nd_argument(p_n, p_c, p_s, pr["proc-size"].as<std::string>());
    p_f = pr["filter-dim"].as<int>();
    conv_fwd_algo = pr["conv-fwd-algo"].as<std::string>();
    conv_bwd_data_algo = pr["conv-bwd-data-algo"].as<std::string>();
    conv_bwd_filter_algo = pr["conv-bwd-filter-algo"].as<std::string>();
    pooling_mode = pr["pooling-mode"].as<std::string>();
    backend = pr["backend"].as<std::string>();
    { // mode
      std::string m = pr["mode"].as<std::string>();
      if (m == "SIMPLE") {
        mode = mode_t::SIMPLE;
      } else {
        mode = mode_t::NORMAL;
      }
    }
    halo_exchange_method = distconv::GetHaloExchangeMethod(
        pr["halo-exchange-method"].as<std::string>());
    { // shuffle-method
      std::string method = pr["shuffle-method"].as<std::string>();
      if (method == "MPI") {
        shuffle_method = distconv::ShuffleMethod::MPI;
      } else if (method == "AL") {
        shuffle_method = distconv::ShuffleMethod::AL;
#ifdef DISTCONV_HAS_P2P
      } else if (method == "P2P") {
        shuffle_method = distconv::ShuffleMethod::P2P;
      } else if (method == "HYBRID") {
        shuffle_method = distconv::ShuffleMethod::HYBRID;
#endif // DISTCONV_HAS_P2P
      } else {
        std::cerr << "Unknown method name for shuffling\n";
        abort();
      }
    }

    {  // Channel/filter parallelism algorithm.
      std::string algo = pr["chanfilt-algo"].as<std::string>();
      if (algo == "NONE") {
        chanfilt_algo = distconv::ChannelParallelismAlgorithm::NONE;
      } else if (algo == "AUTO") {
        chanfilt_algo = distconv::ChannelParallelismAlgorithm::AUTO;
      } else if (algo == "X") {
        chanfilt_algo = distconv::ChannelParallelismAlgorithm::X;
      } else if (algo == "Y") {
        chanfilt_algo = distconv::ChannelParallelismAlgorithm::Y;
      } else if (algo == "W") {
        chanfilt_algo = distconv::ChannelParallelismAlgorithm::W;
        if (p_f == 0) {
          std::cerr << "Must specify --filter-dim for stationary-w\n";
          abort();
        }
      } else {
        std::cerr << "Unknown algorithm name for channel/filter algorithm\n";
        abort();
      }
    }

    batchnorm_impl = distconv::GetBatchnormImpl(
        pr["bn-impl"].as<std::string>());

    // The following arguments are optional.
    if (pr.count("num-dims") > 0) {
      // The --num-dims argument should have been parsed to be NSD.
      assert_eq(NSD, pr["num-dims"].as<int>());
    }
    if (pr.count("conv-algo") > 0) {
      conv_fwd_algo = pr["conv-algo"].as<std::string>();
      conv_bwd_data_algo = pr["conv-algo"].as<std::string>();
      conv_bwd_filter_algo = pr["conv-algo"].as<std::string>();
    }
    if (pr.count("dump-input") > 0) {
      dump_input = true;
    }
    if (pr.count("dump-output") > 0) {
      dump_output = true;
    }
    if (pr.count("dump") > 0) {
      dump_input = true;
      dump_output = true;
    }
    if (pr.count("dump-binary") > 0) {
      dump_binary = true;
    }
    if (pr.count("data-type") > 0) {
      std::string type_name = pr["data-type"].as<std::string>();
      if (type_name == "float") {
        data_type = BenchmarkDataType::FLOAT;
      } else if (type_name == "double") {
        data_type = BenchmarkDataType::DOUBLE;
      } else if (type_name == "half") {
        data_type = BenchmarkDataType::HALF;
      } else {
        std::cerr << "Unknown data type\n";
        abort();
      }
    }
    if (pr.count("profile") > 0) {
      profiling = pr["profile"].as<bool>();
    }
    if (pr.count("nvtx") > 0) {
      nvtx_marking = pr["nvtx"].as<bool>();
    }
    if (use_padding) {
      assert_eq((unsigned int) NSD, f_s.size());
      assert_eq((unsigned int) NSD, dilations.size());
      pads.resize(NSD);
      for(int s = 0; s < NSD; s++) {
        pads[s] = (get_dilated_filter_size(f_s[s], dilations[s]) - 1) / 2;
        assert0((get_dilated_filter_size(f_s[s], dilations[s]) - 1) % 2);
      }
    } else {
      pads = int_vector(NSD, 0);
    }
    if (backend == "Ref") {
      assert_always(data_type != BenchmarkDataType::HALF);
    }
    if (pr.count("overlap") > 0) {
      overlap_halo_exchange = pr["overlap"].as<bool>();
    }
    testing = pr["testing"].as<bool>();
    if (testing) {
      warming_up_count = 0;
      run_count = 1;
    }
    if (pr.count("deterministic") > 0 || testing) {
      deterministic = true;
      conv_fwd_algo = "DETERMINISTIC";
      conv_bwd_data_algo = "DETERMINISTIC";
      conv_bwd_filter_algo = "DETERMINISTIC";
    }
    if (pr.count("skip-allreduce") > 0) {
      skip_weight_allreduce = true;
    }
    if (pr.count("skip-halo-exchange") > 0) {
      skip_halo_exchange = true;
    }
    if (pr.count("skip-chanfilt-comm") > 0) {
      skip_chanfilt_comm = true;
    }
    spin_time_ms = pr["spin-time"].as<int>();
    if (pr.count("host") > 0) {
      host = true;
    }
    if (pr.count("global-stat") > 0) {
      global_stat = true;
    }
    if (pr.count("deconv") > 0) {
      deconv = true;
    }

    assert_num_spatial_dims();
  }

  // Check if the numbers of spatial dimensions are correct.
  void assert_num_spatial_dims() const {
    const std::vector<const int_vector *> svecs =
        {&i_s, &f_s, &p_s, &pads, &strides, &dilations};
    for(auto i : svecs)
      assert_eq((unsigned int) NSD, i->size());
  }

  // Return the number of spatial dimensions.
  int get_num_spatial_dims() const {
    assert_num_spatial_dims();
    return NSD;
  }

  std::ostream &print(std::ostream &os) const {
    const auto reverse_and_join_array =
        [](const int_vector v) {
          return distconv::util::join_xd_array(distconv::util::reverse(v));
        };

    std::stringstream ss;
    ss << "input dims: " << i_n << "x" << i_c << "x" << reverse_and_join_array(i_s)
       << ", filter dims: " << (f_k != -1 ? std::to_string(f_k) + "x" : "")
       <<  reverse_and_join_array(f_s)
       << ", padding: " << reverse_and_join_array(pads)
       << ", stride: " << reverse_and_join_array(strides)
       << ", dilation: " << reverse_and_join_array(dilations)
       << ", group count: " << num_groups
       << ", proc dims: " << p_n << "x" << p_c << "x" << reverse_and_join_array(p_s)
       << ", proc F dim: " << p_f
       << ", backend: " << backend
       << ", fwd algorithm: " << conv_fwd_algo
       << ", bwd data algorithm: " << conv_bwd_data_algo
       << ", bwd filter algorithm: " << conv_bwd_filter_algo
       << ", halo exchange method: " << halo_exchange_method
       << ", shuffle method: " << shuffle_method
       << ", pooling mode: " << pooling_mode
       << ", overlap halo exchange: " << overlap_halo_exchange
       << ", testing: " << testing
       << ", global stat: " << global_stat
       << ", deconv: " << deconv
       << std::endl;
    return os << ss.str();
  }

  std::ostream &print_as_row(std::ostream &os) const {
    const auto reverse_and_join_array =
        [](const int_vector v) {
          return distconv::util::join_spaced_array(distconv::util::reverse(v));
        };

    os << i_n << " " << i_c << " " << reverse_and_join_array(i_s) << " "
       << f_k << " " << reverse_and_join_array(f_s) << " "
       << reverse_and_join_array(pads) << " "
       << reverse_and_join_array(strides) << " "
       << reverse_and_join_array(dilations) << " "
       << num_groups << " "
       << p_n << " " << p_c << " " << reverse_and_join_array(p_s) << " "
       << p_f << " "
       << backend << " "
       << conv_fwd_algo << " " << conv_bwd_data_algo
       << " " << conv_bwd_filter_algo
       << " " << pooling_mode
       << " " << halo_exchange_method
       << " " << overlap_halo_exchange
       << " " << shuffle_method;
    return os;
  }

  // Parse `arg` as a space-separated int vector and substitute the
  // elements to `spatials`.
  static void substitute_nd_argument(int_vector &spatials,
                                     const std::string arg) {
    const int_vector size = distconv::util::split_spaced_array<int>(arg);
    spatials = distconv::util::reverse(int_vector(size.begin(), size.end()));
  }

  // Parse `arg` as a space-separated int vector and substitute the
  // first element to `k`, and the others to `spatials`.
  static void substitute_nd_argument(int &k,
                                     int_vector &spatials,
                                     const std::string arg) {
    const int_vector size = distconv::util::split_spaced_array<int>(arg);
    assert_always(size.size() > 1);
    k = size[0];
    spatials = distconv::util::reverse(int_vector(size.begin()+1, size.end()));
  }

  // Parse `arg` as a space-separated int vector and substitute the
  // first element to `n`, the second to `c`, and the others to `spatials`.
  static void substitute_nd_argument(int &n, int &c,
                                     int_vector &spatials,
                                     const std::string arg) {
    const int_vector size = distconv::util::split_spaced_array<int>(arg);
    assert_always(size.size() > 2);
    n = size[0];
    c = size[1];
    spatials = distconv::util::reverse(int_vector(size.begin()+2, size.end()));
  }
};

template <int NSD>
std::ostream &operator<<(std::ostream &os, BenchmarkConfig<NSD> &cfg) {
  return cfg.print(os);
}

template <int NSD>
inline BenchmarkConfig<NSD> process_opt(int argc, char *argv[], int pid,
                                        const bool is_conv) {
  // Human-readable notations of image/filter shapes
  const std::string shape_notation = std::string(" <N,C") + (NSD == 3 ? ",D" : "") + ",H,W>";
  const std::string filter_shape_notation = std::string(" <K") + (NSD == 3 ? ",D" : "") + ",H,W>";

  // Return a comma-separated `{[n], [c], s, ..., s}` vector.
  // Each of `n` and `c` is available only if it is a positive.
  const auto create_default_size =
      [](const int n, const int c, const int s) {
        int_vector v;
        if(n > 0) v.push_back(n);
        if(c > 0) v.push_back(c);
        for(int i = 0; i < NSD; i++)
          v.push_back(s);
        return distconv::util::join_array(v, ",");
      };

  const auto default_image_size  = create_default_size(8, 16, 32);
  const auto default_filter_size = create_default_size(0, is_conv ? 16 : 0, 3);
  const auto default_proc_size   = create_default_size(1, 1, 1);
  const auto default_strides     = create_default_size(0, 0, 1);
  const auto default_dilations   = create_default_size(0, 0, 1);

  cxxopts::Options cmd_opts(argv[0], "Distributed Convolution Benchmark");
  cmd_opts.add_options()
      ("r,num-runs", "Number of runs", cxxopts::value<int>()->default_value("5"))
      ("num-warmup-runs", "Number of warming-up runs", cxxopts::value<int>()->default_value("5"))
      ("o,output-file", "Save performance profile to file", cxxopts::value<std::string>()->default_value("results"))
      ("image-size", "Image size" + shape_notation, cxxopts::value<std::string>()->default_value(default_image_size))
      ("filter-size", "Filter size" + filter_shape_notation, cxxopts::value<std::string>()->default_value(default_filter_size))
      ("no-padding", "Does not use padding", cxxopts::value<bool>()->default_value("false"))
      ("use-bias", "Use bias", cxxopts::value<bool>()->default_value("false"))
      ("strides", "Vertical and horizontal stride", cxxopts::value<std::string>()->default_value(default_strides))
      ("dilations", "Vertical and horizontal dilation", cxxopts::value<std::string>()->default_value(default_dilations))
      ("num-groups", "Number of convolution groups", cxxopts::value<int>()->default_value("1"))
      ("proc-size", "Process grid size" + shape_notation, cxxopts::value<std::string>()->default_value(default_proc_size))
      ("filter-dim", "Process grid filter dimension", cxxopts::value<int>()->default_value("0"))
      ("a,conv-fwd-algo", "Convolution fwd algorithm", cxxopts::value<std::string>()->default_value("DEFAULT"))
      ("g,conv-bwd-data-algo", "Convolution bwd data algorithm", cxxopts::value<std::string>()->default_value("DEFAULT"))
      ("k,conv-bwd-filter-algo", "Convolution bwd filter algorithm", cxxopts::value<std::string>()->default_value("DEFAULT"))
      ("pooling-mode", "Pooling mode", cxxopts::value<std::string>()->default_value("MAX"))
      ("b,backend", "Convolution backend", cxxopts::value<std::string>()->default_value("CUDNN"))
      ("data-type", "Data type", cxxopts::value<std::string>()->default_value("float"))
      ("mode", "Test mode", cxxopts::value<std::string>()->default_value("NORMAL"))
      ("halo-exchange-method", "Halo exchange method", cxxopts::value<std::string>()->default_value("AL"))
      ("shuffle-method", "Shuffle method", cxxopts::value<std::string>()->default_value("AL"))
      ("bn-impl", "Batchnorm implementation", cxxopts::value<std::string>()->default_value("MPI"))

      ("num-dims", "Number of spatial dimensions", cxxopts::value<int>())
      ("conv-algo", "Convolution algorithm", cxxopts::value<std::string>())
      ("i,dump-input", "Dump input tensors")
      ("d,dump-output", "Dump output tensors")
      ("dump", "Dump input and output tensors")
      ("dump-binary", "Dump tensor in a binary format")
      ("profile", "Enable detailed profiling")
      ("nvtx", "Enable NVTX-based region marking")
      ("overlap", "Overlap halo exchanges")
      ("deterministic", "Use deterministic algoirthms")
      ("skip-allreduce", "Skip allreduces of weights")
      ("skip-halo-exchange", "Skip halo exchange")
      ("testing", "Run benchmarks as tests",
       cxxopts::value<bool>()->default_value("false"))
      ("chanfilt-algo", "Channel/filter parallelism algorithm", cxxopts::value<std::string>()->default_value("NONE"))
      ("skip-chanfilt-comm", "Skip channel/filter communication")
      ("spin-time", "Mili seconds to spin", cxxopts::value<int>()->default_value("100"))
      ("host", "Run benchmark on host (only applicable ot shuffle_benchmark")
      ("global-stat", "Use global statistics with batchnorm")
      ("deconv", "Runs deconvolutions instead of normal convolutions")
      ("help", "Print help")
      ;
  auto result = cmd_opts.parse(argc, argv);
  if (result.count("help")) {
    if (pid == 0) {
      std::cout << cmd_opts.help() << "\n";
    }
    DISTCONV_CHECK_MPI(MPI_Finalize());
    exit(0);
  }
  BenchmarkConfig<NSD> o(result, is_conv);
  return o;
}

int parse_num_dims(int argc, char *argv[]) {
  // Parse only the --num-dims argument. This have to be done before
  // `process_opt` since it requires the `NSD` template parameter,
  // which is obtained here.
  cxxopts::Options cmd_opts(argv[0], "Distributed Convolution Benchmark");
  cmd_opts
      .allow_unrecognised_options()
      .add_options()
      ("num-dims", "Number of spatial dimensions", cxxopts::value<int>()->default_value("2"));
  const auto result = cmd_opts.parse(argc,  argv);
  const int nsd = result["num-dims"].as<int>();
  return nsd;
}

} // namespace distconv_benchmark
