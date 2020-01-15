#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <algorithm>

#include <cuda.h>
#include <cudnn.h>

#include "distconv/util/util.hpp"
#include "distconv/util/stopwatch.h"
#include "distconv/util/util_cudnn.hpp"
#include "distconv/util/cxxopts.hpp"
#include "benchmark_common.hpp"

using namespace distconv;
using distconv_benchmark::BenchmarkDataType;
using distconv_benchmark::BenchmarkConfig;

//#define CUDNN_DATA_REAL CUDNN_DATA_FLOAT

cudnnHandle_t cudnn_h;
cudaStream_t stream;
//const REAL ONE(1.0);
//const REAL ZERO(0.0);
#define BIAS_INIT (0.01)

#if 0
class Opts {
 public:
  constexpr static int max_num_spatial_dims = 3;
  int num_spatial_dims;
  std::string tensor_fmt;

  int num_samples;
  int num_channels;
  std::vector<int> spatial_dims;
  int num_filters;
  std::vector<int> filter_dims;

  std::vector<int> strides;
  std::vector<int> dilations;

  bool use_padding;
  bool use_bias;

  int warming_up_count;
  int run_count;

  cudnnConvolutionFwdAlgo_t conv_fwd_algo;
  cudnnConvolutionBwdDataAlgo_t conv_bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t conv_bwd_filter_algo;

  std::string output_file;

  bool dump_input;
  bool dump_output;
  bool dump_binary;

  BenchmarkDataType data_type;

  Opts(): num_spatial_dims(max_num_spatial_dims),
          tensor_fmt("NCHW"),
          num_samples(8), num_channels(16),
          spatial_dims(max_num_spatial_dims, 32),
          num_filters(16),
          filter_dims(max_num_spatial_dims, 3),
          strides(max_num_spatial_dims, 1),
          dilations(max_num_spatial_dims, 1),
          use_padding(true), use_bias(false),
          warming_up_count(2), run_count(5),
          conv_fwd_algo(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM),
          conv_bwd_data_algo(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1),
          conv_bwd_filter_algo(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1),
          output_file(""),
          dump_input(false),
          dump_output(false),
          dump_binary(false),
          data_type(BenchmarkDataType::FLOAT) {}

  Opts(const cxxopts::ParseResult &pr): Opts() {
    if (pr.count("num-dims") > 0) {
      num_spatial_dims = pr["num-dims"].as<int>();
    } else {
      num_spatial_dims = 2;
    }
    spatial_dims.resize(num_spatial_dims);
    strides.resize(num_spatial_dims);
    dilations.resize(num_spatial_dims);
    filter_dims.resize(num_spatial_dims);
    if (pr.count("width") > 0) {
      spatial_dims[num_spatial_dims-1] = pr["width"].as<int>();
    }
    if (pr.count("height") > 0) {
      spatial_dims[num_spatial_dims-2] = pr["height"].as<int>();
    }
    if (num_spatial_dims >= 3) {
      if (pr.count("depth") > 0) {
        spatial_dims[num_spatial_dims-3] = pr["depth"].as<int>();
      }
    }
    if (pr.count("num-samples") > 0) {
      num_samples = pr["num-samples"].as<int>();
    }
    if (pr.count("num-channels") > 0) {
      num_channels = pr["num-channels"].as<int>();
    }
    if (pr.count("filter-width") > 0) {
      filter_dims[num_spatial_dims-1] = pr["filter-width"].as<int>();
    }
    if (pr.count("filter-height") > 0) {
      filter_dims[num_spatial_dims-2] = pr["filter-height"].as<int>();
    }
    if (num_spatial_dims >= 3) {
      if (pr.count("filter-depth") > 0) {
        filter_dims[num_spatial_dims-3] = pr["filter-depth"].as<int>();
      }
    }
    if (pr.count("num-filters") > 0) {
      num_filters = pr["num-filters"].as<int>();
    }
    if (pr.count("stride") > 0) {
      std::fill(strides.begin(), strides.end(),
                pr["stride"].as<int>());
    } else {
      if (pr.count("stride-width") > 0) {
        strides[num_spatial_dims-1] = pr["stride-width"].as<int>();
      }
      if (pr.count("stride-height") > 0) {
        strides[num_spatial_dims-2] = pr["stride-height"].as<int>();
      }
      if (num_spatial_dims >= 3) {
        if (pr.count("stride-depth") > 0) {
          strides[num_spatial_dims-3] = pr["stride-depth"].as<int>();
        }
      }
    }
    if (pr.count("dilation") > 0) {
      std::fill(dilations.begin(), dilations.end(),
                pr["dilation"].as<int>());
    } else {
      if (pr.count("dilation-width") > 0) {
        dilations[num_spatial_dims-1] = pr["dilation-width"].as<int>();
      }
      if (pr.count("dilation-height") > 0) {
        dilations[num_spatial_dims-2] = pr["dilation-height"].as<int>();
      }
      if (num_spatial_dims >= 3) {
        if (pr.count("dilation-depth") > 0) {
          dilations[num_spatial_dims-3] = pr["dilation-depth"].as<int>();
        }
      }
    }
    if (pr.count("no-padding") > 0) {
      use_padding = !pr["no-padding"].as<bool>();
    }
    if (pr.count("use-bias") > 0) {
      use_bias = pr["use-bias"].as<bool>();
    }
    if (pr.count("conv-fwd-algo") > 0) {
      conv_fwd_algo = util::CUDNNConvolutionFwdAlgorithms::get_algo(
          pr["conv-fwd-algo"].as<std::string>());
    }
    if (pr.count("conv-bwd-data-algo") > 0) {
      conv_bwd_data_algo = util::CUDNNConvolutionBwdDataAlgorithms::get_algo(
          pr["conv-bwd-data-algo"].as<std::string>());
    }
    if (pr.count("conv-bwd-filter-algo") > 0) {
      conv_bwd_filter_algo = util::CUDNNConvolutionBwdFilterAlgorithms::get_algo(
          pr["conv-bwd-filter-algo"].as<std::string>());
    }
    if (pr.count("conv-algo") > 0) {
      conv_fwd_algo = util::CUDNNConvolutionFwdAlgorithms::get_algo(
          pr["conv-algo"].as<std::string>());
      conv_bwd_data_algo = util::CUDNNConvolutionBwdDataAlgorithms::get_algo(
          pr["conv-algo"].as<std::string>());
      conv_bwd_filter_algo = util::CUDNNConvolutionBwdFilterAlgorithms::get_algo(
          pr["conv-algo"].as<std::string>());
    }
    if (pr.count("num-runs") > 0) {
      run_count = pr["num-runs"].as<int>();
    }
    if (pr.count("output-file") > 0) {
      output_file = pr["output-file"].as<std::string>();
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
      dump_binary = pr["dump-binary"].as<bool>();
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
  }

  std::ostream &print(std::ostream &os) {
    return os << "Number of samples: " << num_samples
              << ", number of channels: " << num_channels
              << ", sample dims: "
              << util::tostring(spatial_dims.begin(), spatial_dims.end())
              << ", number of filters: " << num_filters
              << ", filter dims: "
              << util::tostring(filter_dims.begin(), filter_dims.end())
              << ", strides: "
              << util::tostring(strides.begin(), strides.end())
              << ", dilations: "
              << util::tostring(dilations.begin(), dilations.end())
              << ", padding: " << use_padding
              << ", conv fwd algorithm: " << conv_fwd_algo
              << ", conv bwd data algorithm: " << conv_bwd_data_algo
              << ", conv bwd filter algorithm: " << conv_bwd_filter_algo;
  }

  std::ostream &print_as_row(std::ostream &os) {
    os << num_samples << " " << num_channels << " "
       << util::join(" ", spatial_dims.begin(), spatial_dims.end())
       << " " << num_filters << " "
       << util::join(" ", filter_dims.begin(), filter_dims.end())
       << " "
       << util::join(" ", strides.begin(), strides.end())
       << " "
       << util::join(" ", dilations.begin(), dilations.end())
       << " " << conv_fwd_algo << " " << conv_bwd_data_algo << " "
       << conv_bwd_filter_algo;
    return os;
  }
};

std::ostream &operator<<(std::ostream &os, Opts &o) {
  return o.print(os);
}
#endif

template <int NSD>
class Profile {
 public:
  const BenchmarkConfig<NSD> &m_cfg;
  std::vector<float> conv_fwd_time;
  std::vector<float> conv_bwd_data_time;
  std::vector<float> conv_bwd_filter_time;
  std::vector<float> conv_bwd_bias_time;

  Profile(const BenchmarkConfig<NSD> &cfg): m_cfg(cfg) {}

  std::ostream &print_as_row(std::ostream &os) {
    for (size_t i = 0; i < conv_fwd_time.size(); ++i) {
      std::stringstream ss;
      m_cfg.print_as_row(ss) << " " << conv_fwd_time[i] << " "
                             << conv_bwd_data_time[i] << " "
                             << conv_bwd_filter_time[i];
      if (i < conv_bwd_bias_time.size()) {
        ss << " " << conv_bwd_bias_time[i];
      }
      os << ss.str() << std::endl;
    }
    return os;
  }

  void print_summary(std::ostream &os) {
    using namespace distconv_benchmark;
    std::cout << "Forward mean: " << get_mean(conv_fwd_time)
              << ", median: " << get_median(conv_fwd_time)
              << ", min: " << get_min(conv_fwd_time)
              << ", max: " << get_max(conv_fwd_time)
              << "\n";
    std::cout << "Backward data mean: "
              << get_mean(conv_bwd_data_time)
              << ", median: " << get_median(conv_bwd_data_time)
              << ", min: " << get_min(conv_bwd_data_time)
              << ", max: " << get_max(conv_bwd_data_time)
              << "\n";
    std::cout << "Backward filter mean: "
              << get_mean(conv_bwd_filter_time)
              << ", median: " << get_median(conv_bwd_filter_time)
              << ", min: " << get_min(conv_bwd_filter_time)
              << ", max: " << get_max(conv_bwd_filter_time)
              << "\n";
    if (m_cfg.use_bias) {
      std::cout << "Backward bias mean: "
                << get_mean(conv_bwd_bias_time)
                << ", median: " << get_median(conv_bwd_bias_time)
                << ", min: " << get_min(conv_bwd_bias_time)
                << ", max: " << get_max(conv_bwd_bias_time)
                << "\n";
    }
  }
};

template <typename REAL>
class Data {
 public:
  REAL *m_x = nullptr;
  REAL *m_y = nullptr;
  REAL *m_f = nullptr;
  REAL *m_b = nullptr;
  REAL *m_dx = nullptr;
  REAL *m_dy = nullptr;
  REAL *m_df = nullptr;
  REAL *m_db = nullptr;
  cudnnTensorDescriptor_t m_x_d;
  cudnnTensorDescriptor_t m_y_d;
  cudnnFilterDescriptor_t m_f_d;
  cudnnTensorDescriptor_t m_b_d;
  cudnnTensorDescriptor_t m_dx_d;
  cudnnTensorDescriptor_t m_dy_d;
  cudnnFilterDescriptor_t m_df_d;
  cudnnTensorDescriptor_t m_db_d;

  template <int NSD>
  Data(const BenchmarkConfig<NSD> &cfg, REAL *x, REAL *y, REAL *f, REAL *b,
       REAL *dx, REAL *dy, REAL *df, REAL *db):
      m_x(x), m_y(y), m_f(f), m_b(b), m_dx(dx), m_dy(dy), m_df(df), m_db(db) {
    int nd = cfg.get_num_spatial_dims() + 2;
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_x_d));
    auto &&x_d_dims = util::get_cudnn_dims(cfg.i_n, cfg.i_c, cfg.i_s);
    auto &&x_d_strides = util::get_cudnn_strides(cfg.i_n, cfg.i_c,
                                                 cfg.i_s, "NCHW");
    DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        m_x_d, util::get_cudnn_type<REAL>(), nd,
        x_d_dims.data(), x_d_strides.data()));
    std::cout << "x_d: " << util::tostring(m_x_d) << "\n";

    DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_f_d));
    auto &&f_d_dims = util::get_cudnn_dims(
        cfg.f_k, cfg.i_c, cfg.f_s);
    DISTCONV_CHECK_CUDNN(cudnnSetFilterNdDescriptor(
        m_f_d, util::get_cudnn_type<REAL>(), CUDNN_TENSOR_NCHW, nd,
        f_d_dims.data()));
    std::cout << "f_d: " << util::tostring(m_f_d) << "\n";

    std::vector<int> output_spatial_dims;
    for (int i = 0; i < cfg.get_num_spatial_dims(); ++i) {
      int odim = cfg.i_s[i];
      if (!cfg.use_padding) {
        odim -= (distconv_benchmark::get_dilated_filter_size(
            cfg.f_s[i], cfg.dilations[i]) - 1);
      }
      output_spatial_dims.push_back((odim + cfg.strides[i] - 1) / cfg.strides[i]);
    }

    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_y_d));
    auto &&y_d_dims = util::get_cudnn_dims(cfg.i_n, cfg.f_k,
                                           output_spatial_dims);
    auto &&y_d_strides = util::get_cudnn_strides(
        cfg.i_n, cfg.f_k, output_spatial_dims, "NCHW");
    DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        m_y_d, util::get_cudnn_type<REAL>(), nd,
        y_d_dims.data(), y_d_strides.data()));
    std::cout << "y_d: " << util::tostring(m_y_d) << "\n";

    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_dx_d));
    DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        m_dx_d, util::get_cudnn_type<REAL>(), nd,
        x_d_dims.data(), x_d_strides.data()));
    std::cout << "dx_d: " << util::tostring(m_dx_d) << "\n";

    DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_df_d));
    DISTCONV_CHECK_CUDNN(cudnnSetFilterNdDescriptor(
        m_df_d, util::get_cudnn_type<REAL>(), CUDNN_TENSOR_NCHW, nd,
        f_d_dims.data()));
    std::cout << "df_d: " << util::tostring(m_df_d) << "\n";

    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_dy_d));
    DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
        m_dy_d, util::get_cudnn_type<REAL>(), nd,
        y_d_dims.data(), y_d_strides.data()));
        std::cout << "dy_d: " << util::tostring(m_dy_d) << "\n";

    if (cfg.use_bias) {
      std::vector<int> bias_dims(nd, 1);
      bias_dims[1] = cfg.f_k;
      std::vector<int> bias_strides(nd, 1);
      bias_strides[0] = cfg.f_k;
      DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_b_d));
      DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
          m_b_d, util::get_cudnn_type<REAL>(), nd,
          bias_dims.data(), bias_strides.data()));
      std::cout << "b_d: " << util::tostring(m_b_d) << "\n";
      DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_db_d));
      DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
          m_db_d, util::get_cudnn_type<REAL>(), nd,
          bias_dims.data(), bias_strides.data()));
      std::cout << "db_d: " << util::tostring(m_db_d) << "\n";
    }
  }
};

#if 0
Opts process_opt(int argc, char *argv[]) {
  cxxopts::Options cmd_opts("cudnn_benchmark", "CUDNN Benchmark");
  cmd_opts.add_options()
      ("r,num-runs", "Number of runs", cxxopts::value<int>())
      ("o,output-file", "Save performance profile to file", cxxopts::value<std::string>())
      ("num-dims", "Number of spatial dimensions", cxxopts::value<int>())
      ("h,height", "Sample height", cxxopts::value<int>())
      ("w,width", "Sample width", cxxopts::value<int>())
      ("d,depth", "Sample depth", cxxopts::value<int>())
      ("n,num-samples", "Number of samples", cxxopts::value<int>())
      ("c,num-channels", "Number of channels", cxxopts::value<int>())
      ("filter-height", "Filter height", cxxopts::value<int>())
      ("filter-width", "Filter width", cxxopts::value<int>())
      ("filter-depth", "Filter depth", cxxopts::value<int>())
      ("m,num-filters", "Number of filters", cxxopts::value<int>())
      ("stride-height", "Vertical filter stride", cxxopts::value<int>())
      ("stride-width", "Horizontal filter stride", cxxopts::value<int>())
      ("stride", "Vertical and hotrizontal stride", cxxopts::value<int>())
      ("dilation-height", "Vertical dilation", cxxopts::value<int>())
      ("dilation-width", "Horizontal dilation", cxxopts::value<int>())
      ("dilation", "Vertical and horizontal dilation", cxxopts::value<int>())
      ("no-padding", "Does not use padding", cxxopts::value<bool>())
      ("use-bias", "Use bias", cxxopts::value<bool>())
      ("conv-fwd-algo", "Convolution fwd algorithm",
       cxxopts::value<std::string>())
      ("conv-bwd-data-algo", "Convolution bwd data algorithm",
       cxxopts::value<std::string>())
      ("conv-bwd-filter-algo", "Convolution bwd filter algorithm",
       cxxopts::value<std::string>())
      ("conv-algo", "Convolution algorithm", cxxopts::value<std::string>())
      ("dump-input", "Dump input tensors")
      ("dump-output", "Dump output tensors")
      ("dump", "Dump input and output tensors")
      ("dump-binary", "Dump binary tensors")
      ("data-type", "Data type", cxxopts::value<std::string>())
      ("help", "Print help")
      ;
  auto result = cmd_opts.parse(argc, argv);
  if (result.count("help")) {
    std::cout << cmd_opts.help() << "\n";
    exit(0);
  }
  Opts o(result);
  return o;
}
#endif

size_t calc_len(int n, int c, const std::vector<int> &spatial_dims) {
  size_t s = n * c;
  s *= std::accumulate(spatial_dims.begin(), spatial_dims.end(),
                       1, std::multiplies<int>());
  return s;
}

template <typename REAL>
REAL *make_tensor(int n, int c, const std::vector<int> &spatial_dims) {
  void *ptr;
  size_t s = calc_len(n, c, spatial_dims);
  DISTCONV_CHECK_CUDA(cudaMalloc(&ptr, sizeof(REAL) * s));
  DISTCONV_CHECK_CUDA(cudaMemset(ptr, 0, sizeof(REAL) * s));
  return (REAL*)ptr;
}

template <typename REAL>
REAL *make_random_tensor(int n, int c, const std::vector<int> &spatial_dims,
                         unsigned seed) {
  REAL *ptr = make_tensor<REAL>(n, c, spatial_dims);
  assert_always(ptr != nullptr);
  size_t len = calc_len(n, c, spatial_dims);
  std::srand(seed);
  REAL *buf = new REAL[len];
  {
    for (size_t i = 0; i < len; i++) {
      buf[i] = (float)(rand()) / RAND_MAX;
    }
  }
  DISTCONV_CHECK_CUDA(cudaMemcpy(ptr, buf, len * sizeof(REAL),
                                 cudaMemcpyHostToDevice));
  delete[] buf;
  return ptr;
}

template <typename REAL>
REAL *make_constant_tensor(int n, int c, const std::vector<int> &spatial_dims,
                           REAL d) {
  REAL *ptr = make_tensor<REAL>(n, c, spatial_dims);
  assert_always(ptr != nullptr);
  size_t len = calc_len(n, c, spatial_dims);
  REAL *buf = new REAL[len];
  for (size_t i = 0; i < len; i++) {
    buf[i] = d;
  }
  DISTCONV_CHECK_CUDA(cudaMemcpy(ptr, buf, len * sizeof(REAL),
                                 cudaMemcpyHostToDevice));
  delete[] buf;
  return ptr;
}

template <typename REAL>
REAL *make_initialized_tensor(int n, int c, const std::vector<int> &spatial_dims,
                              unsigned seed) {
  REAL *ptr = make_tensor<REAL>(n, c, spatial_dims);
  assert_always(ptr != nullptr);
  size_t len = calc_len(n, c, spatial_dims);
  REAL *buf = new REAL[len];
  distconv_benchmark::Initializer<REAL> init(seed);
  size_t offset = 0;
  int w = spatial_dims[0];
  int h = spatial_dims[1];

  const auto v2s =
    [](const std::vector<int> v) {
      return std::vector<size_t>(v.begin(), v.end());
    };

  for (int i0 = 0; i0 < n; ++i0) {
    for (int i1 = 0; i1 < c; ++i1) {
      if (spatial_dims.size() == 2) {
        for (int i2 = 0; i2 < h; ++i2) {
          for (int i3 = 0; i3 < w; ++i3) {
            const std::vector<int> indices{i3, i2, i1, i0};
            const std::vector<int> dims{w, h, c, n};
            buf[offset] = init.get_initial_value(v2s(indices), v2s(dims));
            ++offset;
          }
        }
      } else if (spatial_dims.size() == 3) {
        int d = spatial_dims[2];
        for (int i2 = 0; i2 < d; ++i2) {
          for (int i3 = 0; i3 < h; ++i3) {
            for (int i4 = 0; i4 < w; ++i4) {
              const std::vector<int> indices{i4, i3, i2, i1, i0};
              const std::vector<int> dims{w, h, d, c, n};
              buf[offset] = init.get_initial_value(v2s(indices), v2s(dims));
              ++offset;
            }
          }
        }
      }
    }
  }
  DISTCONV_CHECK_CUDA(cudaMemcpy(ptr, buf, len * sizeof(REAL),
                                 cudaMemcpyHostToDevice));
  delete[] buf;
  return ptr;
}

template <int NSD, typename REAL>
cudnnConvolutionDescriptor_t get_conv_desc(const BenchmarkConfig<NSD> &cfg) {
  cudnnConvolutionDescriptor_t conv_desc;
  DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
  DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
      conv_desc, cfg.get_num_spatial_dims(),
      cfg.pads.data(), cfg.strides.data(),
      cfg.dilations.data(), CUDNN_CROSS_CORRELATION,
      util::get_cudnn_type<REAL>()));
  // enable Tensor Cores
  DISTCONV_CHECK_CUDNN(cudnnSetConvolutionMathType(
      conv_desc, CUDNN_TENSOR_OP_MATH));
  return conv_desc;
}

template <int NSD, typename REAL>
void setup_workspace(const Data<REAL> &d, cudnnConvolutionDescriptor_t conv_desc,
                     const BenchmarkConfig<NSD> &cfg, size_t &ws_size, void *&ws) {
  size_t f_size, bd_size, bf_size;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
      cudnn_h,
      d.m_x_d,
      d.m_f_d,
      conv_desc,
      d.m_y_d,
      util::CUDNNConvolutionFwdAlgorithms::get_algo(cfg.conv_fwd_algo),
      &f_size));
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
      cudnn_h,
      d.m_f_d,
      d.m_dy_d,
      conv_desc,
      d.m_dx_d,
      util::CUDNNConvolutionBwdDataAlgorithms::get_algo(cfg.conv_bwd_data_algo),
      &bd_size));
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnn_h,
      d.m_x_d,
      d.m_dy_d,
      conv_desc,
      d.m_df_d,
      util::CUDNNConvolutionBwdFilterAlgorithms::get_algo(cfg.conv_bwd_filter_algo),
      &bf_size));
  ws_size = 0;
  ws_size = std::max(f_size, bd_size);
  ws_size = std::max(ws_size, bf_size);
  DISTCONV_CHECK_CUDA(cudaMalloc(&ws, ws_size));
  return;
}

template <int NSD, typename REAL>
void run_forward_convolution(const Data<REAL> &d, cudnnConvolutionDescriptor_t conv_desc,
                             const BenchmarkConfig<NSD> &cfg, size_t ws_size, void *ws) {
  REAL zero(0.0);
  REAL one(1.0);
  DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
      cudnn_h, &one, d.m_x_d, d.m_x,
      d.m_f_d, d.m_f,
      conv_desc,
      util::CUDNNConvolutionFwdAlgorithms::get_algo(cfg.conv_fwd_algo),
      ws, ws_size, &zero,
      d.m_y_d, d.m_y));
    if (cfg.use_bias) {
      DISTCONV_CHECK_CUDNN(cudnnAddTensor(cudnn_h, &one, d.m_b_d, d.m_b, &one,
                                          d.m_y_d, d.m_y));
  }
}

template <int NSD, typename REAL>
void run_backward_data_convolution(const Data<REAL> &d, cudnnConvolutionDescriptor_t conv_desc,
                                   const BenchmarkConfig<NSD> &cfg, size_t ws_size, void *ws) {
  REAL zero(0.0);
  REAL one(1.0);
  DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
      cudnn_h, &one, d.m_f_d, d.m_f,
      d.m_dy_d, d.m_dy,
      conv_desc,
      util::CUDNNConvolutionBwdDataAlgorithms::get_algo(cfg.conv_bwd_data_algo),
      ws, ws_size, &zero,
      d.m_dx_d, d.m_dx));
}

template <int NSD, typename REAL>
void run_backward_filter_convolution(const Data<REAL> &d, cudnnConvolutionDescriptor_t conv_desc,
                                     const BenchmarkConfig<NSD> &cfg, size_t ws_size, void *ws) {
  REAL zero(0.0);
  REAL one(1.0);
  DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
      cudnn_h, &one, d.m_x_d, d.m_x,
      d.m_dy_d, d.m_dy,
      conv_desc,
      util::CUDNNConvolutionBwdFilterAlgorithms::get_algo(cfg.conv_bwd_filter_algo),
      ws, ws_size, &zero,
      d.m_df_d, d.m_df));
}

template <int NSD, typename REAL>
void run_backward_bias_convolution(const Data<REAL> &d, const BenchmarkConfig<NSD> &cfg) {
  REAL zero(0.0);
  REAL one(1.0);
  DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardBias(cudnn_h, &one, d.m_dy_d, d.m_dy,
                                                    &zero, d.m_db_d, d.m_db));
}

template <int NSD, typename REAL>
void measure_forward_convolution(const Data<REAL> &d,
                                 cudnnConvolutionDescriptor_t conv_desc,
                                 size_t ws_size, void *ws,
                                 const BenchmarkConfig<NSD> &cfg,
                                 Profile<NSD> &prof) {
  std::vector<util::Clock> clks(cfg.run_count, stream);
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  if (cfg.warming_up_count > 0) std::cout << "Warming up\n";
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    run_forward_convolution(d, conv_desc, cfg, ws_size, ws);
  }
  std::cout << "Starting " << cfg.run_count << " times of measurement\n";
  for (int i = 0; i < cfg.run_count; ++i) {
    clks[i].start();
    run_forward_convolution(d, conv_desc, cfg, ws_size, ws);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    float elapsed = clks[i].get_time();
    prof.conv_fwd_time.push_back(elapsed);
  }
  std::cout << "Measurement done\n";
  return;
}

template <int NSD, typename REAL>
void measure_backward_data_convolution(const Data<REAL> &d,
                                       cudnnConvolutionDescriptor_t conv_desc,
                                       size_t ws_size, void *ws,
                                       const BenchmarkConfig<NSD> &cfg,
                                       Profile<NSD> &prof) {
  std::vector<util::Clock> clks(cfg.run_count, stream);
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  if (cfg.warming_up_count > 0) std::cout << "Warming up\n";
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    run_backward_data_convolution(d, conv_desc, cfg, ws_size, ws);
  }
  std::cout << "Starting " << cfg.run_count << " times of measurement\n";
  for (int i = 0; i < cfg.run_count; ++i) {
    clks[i].start();
    run_backward_data_convolution(d, conv_desc, cfg, ws_size, ws);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    float elapsed = clks[i].get_time();
    prof.conv_bwd_data_time.push_back(elapsed);
  }
  std::cout << "Measurement done\n";
  return;
}

template <int NSD, typename REAL>
void measure_backward_filter_convolution(const Data<REAL> &d,
                                         cudnnConvolutionDescriptor_t conv_desc,
                                         size_t ws_size, void *ws,
                                         const BenchmarkConfig<NSD> &cfg,
                                         Profile<NSD> &prof) {
  std::vector<util::Clock> clks(cfg.run_count, stream);
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  if (cfg.warming_up_count > 0) std::cout << "Warming up\n";
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    run_backward_filter_convolution(d, conv_desc, cfg, ws_size, ws);
  }
  std::cout << "Starting " << cfg.run_count << " times of measurement\n";
  for (int i = 0; i < cfg.run_count; ++i) {
    clks[i].start();
    run_backward_filter_convolution(d, conv_desc, cfg, ws_size, ws);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    float elapsed = clks[i].get_time();
    prof.conv_bwd_filter_time.push_back(elapsed);
  }
  std::cout << "Measurement done\n";
  return;
}

template <int NSD, typename REAL>
void measure_backward_bias_convolution(const Data<REAL> &d,
                                       const BenchmarkConfig<NSD> &cfg,
                                       Profile<NSD> &prof) {
  std::vector<util::Clock> clks(cfg.run_count, stream);
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  if (cfg.warming_up_count > 0) std::cout << "Warming up\n";
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    run_backward_bias_convolution(d, cfg);
  }
  std::cout << "Starting " << cfg.run_count << " times of measurement\n";
  for (int i = 0; i < cfg.run_count; ++i) {
    clks[i].start();
    run_backward_bias_convolution(d, cfg);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    float elapsed = clks[i].get_time();
    prof.conv_bwd_bias_time.push_back(elapsed);
  }
  std::cout << "Measurement done\n";
  return;
}

std::ostream &print_version_number(std::ostream &os) {
  int version[3];
  cudnnGetProperty(MAJOR_VERSION, &version[0]);
  cudnnGetProperty(MINOR_VERSION, &version[1]);
  cudnnGetProperty(PATCH_LEVEL, &version[2]);
  os << "cuDNN v" << version[0] << "." << version[1] << "." << version[2];
  return os;
}

template <typename REAL>
int dump_tensor(const REAL *t, size_t num_elms,
                const std::string &file_path,
                bool binary) {
  REAL *h = new REAL[num_elms];
  cudaMemcpy(h, t, num_elms * sizeof(REAL), cudaMemcpyDeviceToHost);
  std::ofstream out;
  if (binary) {
    out.open(file_path + ".out", std::ios::out | std::ios::trunc | std::ios::binary);
    out.write((char *)h, num_elms * sizeof(REAL));
  } else {
    out.open(file_path + ".txt", std::ios::out | std::ios::trunc);
    for (size_t i = 0; i < num_elms; ++i) {
      out << h[i] << std::endl;
    }
  }
  out.close();
  delete[] h;
  return 0;
}

template <int NSD, typename REAL>
int run(const BenchmarkConfig<NSD> &cfg) {
  std::srand(0);

  //REAL *input_tensor = make_random_tensor(o.i_n, o.i_c, o.i_h,
  //o.i_w);
  REAL *input_tensor = make_initialized_tensor<REAL>(
      cfg.i_n, cfg.i_c, cfg.i_s,
      distconv_benchmark::input_tensor_seed);
  REAL *d_input_tensor = make_tensor<REAL>(
      cfg.i_n, cfg.i_c, cfg.i_s);
  REAL *filter_tensor = make_initialized_tensor<REAL>(
      cfg.f_k, cfg.i_c, cfg.f_s,
      distconv_benchmark::filter_tensor_seed);
  REAL *d_filter_tensor = make_tensor<REAL>(
      cfg.f_k, cfg.i_c, cfg.f_s);
  std::vector<int> output_spatial_dims;
  for (int i = 0; i < cfg.get_num_spatial_dims(); ++i) {
    int odim = cfg.i_s[i];
    if (!cfg.use_padding) {
      odim -= (distconv_benchmark::get_dilated_filter_size(
          cfg.f_s[i], cfg.dilations[i]) - 1);
    }
    output_spatial_dims.push_back((odim + cfg.strides[i] - 1) / cfg.strides[i]);
  }
  REAL *output_tensor = make_tensor<REAL>(cfg.i_n, cfg.f_k,
                                          output_spatial_dims);
  REAL *d_output_tensor = make_initialized_tensor<REAL>(
      cfg.i_n, cfg.f_k, output_spatial_dims,
      distconv_benchmark::d_output_tensor_seed);
  REAL *bias_tensor = nullptr;
  REAL *d_bias_tensor = nullptr;
  if (cfg.use_bias) {
    bias_tensor = make_constant_tensor<REAL>(
        1, cfg.f_k, std::vector<int>(cfg.get_num_spatial_dims(), 1), BIAS_INIT);
    d_bias_tensor = make_tensor<REAL>(
        1, cfg.f_k, std::vector<int>(cfg.get_num_spatial_dims(), 1));
  }

  Data<REAL> d(cfg, input_tensor, output_tensor, filter_tensor, bias_tensor,
               d_input_tensor, d_output_tensor, d_filter_tensor, d_bias_tensor);

  if (cfg.dump_input) {
    size_t input_tensor_size = calc_len(cfg.i_n, cfg.i_c,
                                        cfg.i_s);
    dump_tensor(input_tensor, input_tensor_size, "input_tensor", cfg.dump_binary);
    size_t filter_tensor_size = calc_len(cfg.f_k, cfg.i_c,
                                         cfg.f_s);
    dump_tensor(filter_tensor, filter_tensor_size, "filter_tensor", cfg.dump_binary);
    size_t output_tensor_size = calc_len(cfg.i_n, cfg.f_k,
                                         output_spatial_dims);
    dump_tensor(d_output_tensor, output_tensor_size, "d_output_tensor",
                cfg.dump_binary);
    if (cfg.use_bias) {
      size_t bias_tensor_size = cfg.f_k;
      dump_tensor(bias_tensor, bias_tensor_size, "bias_tensor", cfg.dump_binary);
    }
  }

  cudnnConvolutionDescriptor_t conv_desc = get_conv_desc<NSD, REAL>(cfg);
  std::cout << "conv_desc: " << util::tostring(conv_desc) << "\n";

  std::cout
      << "Forward algo: "
      << (cfg.deconv ?
          util::CUDNNConvolutionBwdDataAlgorithms::get_real_name(cfg.conv_fwd_algo) :
          util::CUDNNConvolutionFwdAlgorithms::get_real_name(cfg.conv_fwd_algo))
      << "\n"
      << "Bacward data algo: "
      << (cfg.deconv ?
          util::CUDNNConvolutionFwdAlgorithms::get_real_name(cfg.conv_bwd_data_algo) :
          util::CUDNNConvolutionBwdDataAlgorithms::get_real_name(cfg.conv_bwd_data_algo))
      << "\n"
      << "Bacward filter algo: "
      << util::CUDNNConvolutionBwdFilterAlgorithms::get_real_name(cfg.conv_bwd_filter_algo)
      << "\n";

  size_t ws_size;
  void *ws;
  setup_workspace(d, conv_desc, cfg, ws_size, ws);

  Profile<NSD> prof(cfg);
  measure_forward_convolution(d, conv_desc, ws_size, ws, cfg, prof);
  measure_backward_data_convolution(d, conv_desc, ws_size, ws, cfg, prof);
  measure_backward_filter_convolution(d, conv_desc, ws_size, ws, cfg, prof);
  if (cfg.use_bias) {
    measure_backward_bias_convolution(d, cfg, prof);
  }

  std::cout << "Destroying cudnn handle\n";
  DISTCONV_CHECK_CUDNN(cudnnDestroy(cudnn_h));

  prof.print_summary(std::cout);

  std::ostream *output_stream;
  std::ofstream ofs;
  if (cfg.output_file.length() > 0) {
    ofs.open(cfg.output_file);
    output_stream = &ofs;
  } else {
    output_stream = &std::cout;
  }

  prof.print_as_row(*output_stream);

  if (cfg.dump_output) {
    size_t output_tensor_size = calc_len(cfg.i_n, cfg.f_k,
                                         output_spatial_dims);
    dump_tensor(output_tensor, output_tensor_size, "output_tensor", cfg.dump_binary);
    size_t input_tensor_size = calc_len(cfg.i_n, cfg.i_c,
                                        cfg.i_s);
    dump_tensor(d_input_tensor, input_tensor_size, "d_input_tensor", cfg.dump_binary);
    size_t filter_tensor_size = calc_len(cfg.f_k, cfg.i_c, cfg.f_s);
    dump_tensor(d_filter_tensor, filter_tensor_size, "d_filter_tensor", cfg.dump_binary);
    if (cfg.use_bias) {
      size_t bias_tensor_size = cfg.f_k;
      dump_tensor(d_bias_tensor, bias_tensor_size, "d_bias_tensor", cfg.dump_binary);
    }
  }

  std::cout << "Completed\n";
  return 0;
}

template <int NSD>
void run(int argc, char *argv[]) {
  auto cfg = distconv_benchmark::process_opt<NSD>(argc, argv, 0, true);

  if (cfg.data_type == BenchmarkDataType::FLOAT) {
    run<NSD, float>(cfg);
  } else if (cfg.data_type == BenchmarkDataType::DOUBLE) {
    run<NSD, double>(cfg);
  } else if (cfg.data_type == BenchmarkDataType::HALF) {
#ifdef DISTCONV_ENABLE_FP16
    run<NSD, half>(cfg);
#else
    std::cerr << "Error: half precision not supported\n";
    abort();
#endif
  } else {
    std::cerr << "Error: Unknown data type\n";
    abort();
  }
}

int main(int argc, char *argv[]) {
  std::srand(0);
  int dev = 0;
  DISTCONV_CHECK_CUDA(cudaSetDevice(dev));
  DISTCONV_CHECK_CUDNN(cudnnCreate(&cudnn_h));
  print_version_number(std::cout << "Using ") << std::endl;
  cudaStreamCreate(&stream);
  DISTCONV_CHECK_CUDNN(cudnnSetStream(cudnn_h, stream));

  const int nsd = distconv_benchmark::parse_num_dims(argc, argv);

  if(nsd == 2) {
    run<2>(argc, argv);
  } else if(nsd == 3) {
    run<3>(argc, argv);
  } else {
    util::MPIRootPrintStreamError() << "Invalid --num-dims: " << nsd;
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  return 0;
}
