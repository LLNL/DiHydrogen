#pragma once

#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/distconv.hpp"
#include "distconv/util/util_mpi.hpp"
#ifdef DISTCONV_HAS_CUDA
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/util/util_cuda.hpp"
#endif
#ifdef DISTCONV_HAS_CUDNN
#include "distconv/util/util_cudnn.hpp"
#endif
#include "distconv/util/cxxopts.hpp"

namespace test {

using namespace distconv;

class Config {
 public:
  int i_n;
  int i_c;
  int i_h;
  int i_w;

  int f_k;
  int f_h;
  int f_w;

  int pad_h;
  int pad_w;

  bool use_padding;
  bool use_bias;

  int stride_h;
  int stride_w;

  int p_n;
  int p_c;
  int p_h;
  int p_w;

  int warming_up_count;
  int run_count;

  std::string conv_fwd_algo;
  std::string conv_bwd_data_algo;
  std::string conv_bwd_filter_algo;
  std::string backend;

  std::string output_file;

  bool dump_input;
  bool dump_output;

  bool use_global_stat;

  enum mode_t {NORMAL, VALIDATE, CUDNN_VALIDATE};
  mode_t mode;

  Config(): i_n(1), i_c(4), i_h(32), i_w(32),
            f_k(4), f_h(3), f_w(3),
            pad_h(1), pad_w(1),
            use_padding(true),
            use_bias(true),
            stride_h(1), stride_w(1),
            p_n(1), p_c(1), p_h(1), p_w(1),
            warming_up_count(2), run_count(5),
            conv_fwd_algo("DEFAULT"),
            conv_bwd_data_algo("ALGO1"),
            conv_bwd_filter_algo("ALGO1"),
            backend("CUDNN"),
            output_file(""),
            dump_input(false),
            dump_output(false),
            use_global_stat(false),
            mode(mode_t::NORMAL) {}

  Config(const cxxopts::ParseResult &pr): Config() {
    if (pr.count("image-height") > 0) {
      i_h = pr["image-height"].as<int>();
    }
    if (pr.count("image-width") > 0) {
      i_w = pr["image-width"].as<int>();
    }
    if (pr.count("image-size") > 0) {
      i_h = pr["image-size"].as<int>();
      i_w = pr["image-size"].as<int>();
    }
    if (pr.count("num-images") > 0) {
      i_n = pr["num-images"].as<int>();
    }
    if (pr.count("num-channels") > 0) {
      i_c = pr["num-channels"].as<int>();
    }
    if (pr.count("filter-height") > 0) {
      f_h = pr["filter-height"].as<int>();
    }
    if (pr.count("filter-width") > 0) {
      f_w = pr["filter-width"].as<int>();
    }
    if (pr.count("filter-size") > 0) {
      f_w = pr["filter-size"].as<int>();
      f_h = pr["filter-size"].as<int>();
    }
    if (pr.count("num-filters") > 0) {
      f_k = pr["num-filters"].as<int>();
    }
    if (pr.count("no-padding") > 0) {
      use_padding = !pr["no-padding"].as<bool>();
    }
    if (pr.count("use-bias") > 0) {
      use_bias = pr["use-bias"].as<bool>();
    }
    if (pr.count("stride-width") > 0) {
      stride_w = pr["stride-width"].as<int>();
    }
    if (pr.count("stride-height") > 0) {
      stride_h = pr["stride-height"].as<int>();
    }
    if (pr.count("stride") > 0) {
      stride_h = pr["stride"].as<int>();
      stride_w = pr["stride"].as<int>();
    }
    if (pr.count("proc-n") > 0) {
      p_n = pr["proc-n"].as<int>();
    }
    if (pr.count("proc-c") > 0) {
      p_c = pr["proc-c"].as<int>();
    }
    if (pr.count("proc-h") > 0) {
      p_h = pr["proc-h"].as<int>();
    }
    if (pr.count("proc-w") > 0) {
      p_w = pr["proc-w"].as<int>();
    }
    if (pr.count("conv-fwd-algo") > 0) {
      conv_fwd_algo = pr["conv-fwd-algo"].as<std::string>();
    }
    if (pr.count("conv-bwd-data-algo") > 0) {
      conv_bwd_data_algo = pr["conv-bwd-data-algo"].as<std::string>();
    }
    if (pr.count("conv-bwd-filter-algo") > 0) {
      conv_bwd_filter_algo = pr["conv-bwd-filter-algo"].as<std::string>();
    }
    if (pr.count("num-runs") > 0) {
      run_count = pr["num-runs"].as<int>();
    }
    if (pr.count("num-warmup-runs") > 0) {
      warming_up_count = pr["num-warmup-runs"].as<int>();
    }
    if (pr.count("backend") > 0) {
      backend = pr["backend"].as<std::string>();
    }
    if (pr.count("output-file") > 0) {
      output_file = pr["output-file"].as<std::string>();
    }
    if (pr.count("dump-input") > 0) {
      dump_input = pr["dump-input"].as<bool>();
    }
    if (pr.count("dump-output") > 0) {
      dump_output = pr["dump-output"].as<bool>();
    }
    if (pr.count("use-global-stat") > 0) {
      use_global_stat = pr["use-global-stat"].as<bool>();
    }
    if (pr.count("mode") > 0) {
      std::string m = pr["mode"].as<std::string>();
      if (m == "CUDNN_VALIDATE") {
        mode = mode_t::CUDNN_VALIDATE;
      } else if (m == "VALIDATE") {
        mode = mode_t::VALIDATE;
      } else {
        mode = mode_t::NORMAL;
      }
    }
    if (use_padding) {
      pad_h = (f_h - 1) / 2;
      assert_always(((f_h - 1) % 2) == 0);
      pad_w = (f_w - 1) / 2;
      assert_always(((f_w - 1) % 2) == 0);
    } else {
      pad_h = 0;
      pad_w = 0;
    }
  }

  std::ostream &print(std::ostream &os) {
    return os << "input dims: " << i_n << "x" << i_c << "x" << i_h << "x" << i_w
              << ", filter dims: " << f_k << "x" << f_h << "x" << f_w
              << ", padding: " << pad_h << "x" << pad_w
              << ", stride: " << stride_h << "x" << stride_w
              << ", proc dims: " << p_h << "x" << p_w
              << ", backend: " << backend
              << ", fwd algorithm: " << conv_fwd_algo
              << ", bwd data algorithm: " << conv_bwd_data_algo
              << ", bwd filter algorithm: " << conv_bwd_filter_algo
              << "\n";
  }

  std::ostream &print_as_row(std::ostream &os) {
    os << i_n << " " << i_c << " " << i_h << " " << i_w << " "
       << f_k << " " << f_h << " " << f_w << " "
       << pad_h << " " << pad_w << " "
       << stride_h << " " << stride_w << " "
       << p_h << " " << p_w << " "
       << backend << " "
       << conv_fwd_algo << " " << conv_bwd_data_algo << " "
       << conv_bwd_filter_algo;
    return os;
  }
};

std::ostream &operator<<(std::ostream &os, Config &cfg) {
  return cfg.print(os);
}

inline Config process_opt(int argc, char *argv[], int pid) {
  cxxopts::Options cmd_opts(argv[0], "Distributed Convolution Benchmark");
  cmd_opts.add_options()
      ("r,num-runs", "Number of runs", cxxopts::value<int>())
      ("num-warmup-runs", "Number of warming-up runs", cxxopts::value<int>())
      ("o,output-file", "Save performance profile to file", cxxopts::value<std::string>())
      ("h,image-height", "Image height", cxxopts::value<int>())
      ("w,image-width", "Image width", cxxopts::value<int>())
      ("image-size", "Image size", cxxopts::value<int>())
      ("n,num-images", "Number of images", cxxopts::value<int>())
      ("c,num-channels", "Number of channels", cxxopts::value<int>())
      ("s,filter-height", "Filter height", cxxopts::value<int>())
      ("t,filter-width", "Filter width", cxxopts::value<int>())
      ("filter-size", "Filter size", cxxopts::value<int>())
      ("m,num-filters", "Number of filters", cxxopts::value<int>())
      ("no-padding", "Does not use padding", cxxopts::value<bool>())
      ("use-bias", "Use bias", cxxopts::value<bool>())
      ("stride-height", "Vertical filter stride", cxxopts::value<int>())
      ("stride-width", "Horizontal filter stride", cxxopts::value<int>())
      ("stride", "Filter stride", cxxopts::value<int>())
      ("proc-n", "N dimension of process grid", cxxopts::value<int>())
      ("proc-c", "C dimension of process grid", cxxopts::value<int>())
      ("proc-h", "H dimension of process grid", cxxopts::value<int>())
      ("proc-w", "W dimension of process grid", cxxopts::value<int>())
      ("a,conv-fwd-algo", "Convolution fwd algorithm", cxxopts::value<std::string>())
      ("g,conv-bwd-data-algo", "Convolution bwd data algorithm", cxxopts::value<std::string>())
      ("k,conv-bwd-filter-algo", "Convolution bwd filter algorithm", cxxopts::value<std::string>())
      ("b,backend", "Convolution backend", cxxopts::value<std::string>())
      ("i,dump-input", "Dump input tensors")
      ("d,dump-output", "Dump output tensor")
      ("use-global-stat", "Use global statistics in batch normalization")
      ("mode", "Test mode", cxxopts::value<std::string>())
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
  auto o(result);
  return o;
}


template <typename Tensor>
int init_input_tensor(Tensor &t) {
  using data_type = typename Tensor::data_type;

  // Halo region must be set to zero
  t.zero();
  auto local_shape = t.get_local_shape();
  auto *buf = t.get_buffer();
  assert_always(buf);

  size_t buf_size = t.get_local_real_shape().get_size() *
      sizeof(data_type);
  auto *host = (data_type*)malloc(buf_size);
  t.copyout(host);

  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    auto global_index = t.get_global_index(*it);
    typename Tensor::data_type v = global_index[0] + global_index[1];
    host[t.get_local_offset(*it)] = v;
  }

  t.copyin(host);
  return 0;
}

// REFACTORING: Duplicated in benchmark_common.hpp
template <typename REAL>
struct Initializer {
  Initializer(unsigned seed, REAL alpha=0): m_alpha(alpha) {
    std::srand(seed);
    for (int i = 0; i < num_rands; ++i) {
      m_rands[i] = std::rand();
    }
  }
  REAL get_initial_value(size_t n_idx, size_t c_idx, size_t h_idx, size_t w_idx,
                         size_t n_dim, size_t c_dim, size_t h_dim, size_t w_dim) {
    size_t x = n_idx + c_idx + h_idx + w_idx;
    double rand1 = (double)(m_rands[x % num_rands]) / RAND_MAX;
    size_t offset = w_idx + h_idx * w_dim + c_idx * w_dim * h_dim
        + n_idx * w_dim * h_dim * c_dim;
    double rand2 = (double)(m_rands[offset % num_rands]) / RAND_MAX;
    double v = (rand1 + rand2) / 2;
    return static_cast<REAL>(v) + m_alpha;
  }
  static constexpr int num_rands = 1000;
  unsigned m_rands[num_rands];
  REAL m_alpha;
};


template <typename Tensor>
int init_tensor_random(Tensor &t, unsigned seed, typename Tensor::data_type alpha=0) {
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto *buf = t.get_buffer();
  assert_always(buf);

  size_t buf_size = t.get_local_real_shape().get_size() *
      sizeof(data_type);
  auto *host = (data_type*)malloc(buf_size);

  Initializer<typename Tensor::data_type> random_init(seed, alpha);
  const auto global_shape = t.get_shape();
  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    auto global_index = t.get_global_index(*it);
    host[t.get_local_offset(*it)] = random_init.get_initial_value(
        global_index[3], global_index[2], global_index[1], global_index[0],
        global_shape[3], global_shape[2], global_shape[1], global_shape[0]);
  }

  t.copyin(host);
  return 0;
}

template <typename Tensor>
int init_tensor_offset(Tensor &t) {
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto *buf = t.get_buffer();
  assert_always(buf);

  size_t buf_size = t.get_local_real_shape().get_size() *
      sizeof(data_type);
  auto *host = (data_type*)malloc(buf_size);
  t.copyout(host);

  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    host[t.get_local_offset(*it)] = t.get_local_offset(*it);
  }

  t.copyin(host);
  return 0;
}

template <typename Tensor>
int init_tensor_constant(Tensor &t, typename Tensor::data_type x) {
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto *buf = t.get_buffer();
  assert_always(buf);

  size_t buf_size = t.get_local_real_shape().get_size() *
      sizeof(data_type);
  auto *host = (data_type*)malloc(buf_size);
  t.copyout(host);

  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    host[t.get_local_offset(*it)] = x;
  }

  t.copyin(host);
  return 0;
}

template <typename DataType>
int dump_tensor(const tensor::Tensor<DataType, tensor::LocaleProcess,
                tensor::BaseAllocator> &t,
                const std::string &file_path) {
  std::ofstream out;
  out.open(file_path, std::ios::out | std::ios::trunc);
  auto buf = t.get_const_buffer();
  auto shape = t.get_shape();
  for (auto it = shape.index_begin(); it != shape.index_end(); ++it) {
    auto x = buf[t.get_local_offset(*it)];
    out << x << std::endl;
  }
  out.close();
  return 0;
}

template <typename DataType, typename Alloccator>
int dump_tensor(const tensor::Tensor<DataType, tensor::LocaleMPI, Alloccator> &t,
                const std::string &file_path) {
  using TensorProcType = tensor::Tensor<DataType,
                                        tensor::LocaleProcess,
                                        tensor::BaseAllocator>;
  TensorProcType temp_tensor(tensor::LocaleProcess(), 1);

  assert0(tensor::Copy(temp_tensor, t, 0));
  if (t.get_locale().get_rank() == 0) {
    dump_tensor(temp_tensor, file_path);
  }

  return 0;
}

template <typename DataType, typename Alloccator>
int dump_shared_tensor(const tensor::Tensor<DataType, tensor::LocaleMPI,
                       Alloccator> &t,
                       const std::string &file_path,
                       bool binary=false) {
  if (t.get_locale().get_rank() == 0) {
    std::ofstream out;
    DataType *buf = (DataType*)malloc(t.get_size() * sizeof(DataType));
    t.get_data().copyout(buf);
    if (binary) {
      out.open(file_path + ".out",
               std::ios::out | std::ios::trunc | std::ios::binary);
      out.write((char*)buf, t.get_size() * sizeof(DataType));
    } else {
      out.open(file_path + ".txt",
               std::ios::out | std::ios::trunc);
      for (size_t i = 0; i < t.get_size(); ++i) {
        auto x = buf[i];
        out << x << std::endl;
      }
    }
    out.close();
  }
  return 0;
}

} // namespace test
