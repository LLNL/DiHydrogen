#include <vector>

#include "distconv_config.hpp"
#include "distconv_benchmark_common.hpp"
#include "benchmark_common.hpp"

using namespace distconv;

namespace distconv_benchmark {

template <int NSD>
class Profile {
 public:
  std::vector<float> fwd_time;
  std::vector<float> bwd_time;
  BenchmarkConfig<NSD> m_cfg;
  Profile(const BenchmarkConfig<NSD> &cfg): m_cfg(cfg) {}

  std::ostream &print_as_row(std::ostream &os) {
    for (size_t i = 0; i < fwd_time.size(); ++i) {
      m_cfg.print_as_row(os) << " " << fwd_time[i] << " "
                             << bwd_time[i] << std::endl;
    }
    return os;
  }

  void print_summary(std::ostream &os) {
    std::stringstream ss;
    ss << "Forward mean: " << get_mean(fwd_time)
       << ", median: " << get_median(fwd_time)
       << ", min: " << get_min(fwd_time)
       << ", max: " << get_max(fwd_time)
       << std::endl
       << "Backward mean: " << get_mean(bwd_time)
       << ", median: " << get_median(bwd_time)
       << ", min: " << get_min(bwd_time)
       << ", max: " << get_max(bwd_time)
       << std::endl;
    os << ss.str();
  }
};

template <int NSD, typename Backend, typename DataType>
class Data {
 public:
  typename TensorType<Backend, DataType>::type input;
  typename TensorType<Backend, DataType>::type output;
  typename TensorType<Backend, DataType>::type d_input;
  typename TensorType<Backend, DataType>::type d_output;
  const BenchmarkConfig<NSD> m_cfg;

  Data(const BenchmarkConfig<NSD> &cfg, MPI_Comm comm): m_cfg(cfg) {
    using Tensor = typename TensorType<Backend, DataType>::type;

    int pid;
    int np;
    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &np);

    assert_always(std::accumulate(cfg.p_s.begin(),
                                  cfg.p_s.end(),
                                  1,
                                  std::multiplies<int>())
                  * cfg.p_c * cfg.p_n == np);

    assert_always(cfg.p_c == 1 &&
                  "Decomposition over dimension C not supported yet");

    const auto vector_concat =
        [](const int_vector v, const int c, const int n) {
          int_vector cn({c, n});
          cn.insert(cn.begin(), v.begin(), v.end());
          return (const int_vector) cn;
        };

    const auto input_shape  = vector_concat(cfg.i_s, cfg.i_c, cfg.i_n);
    const auto locale_shape = vector_concat(cfg.p_s, cfg.p_c, cfg.p_n);
    const auto window = cfg.f_s;
    const auto strides = cfg.strides;
    const auto pads = cfg.pads;
    const int_vector dilations(cfg.get_num_spatial_dims(), 1);
    input = create_input_tensor<Tensor>(
        input_shape, locale_shape, window, strides,
        dilations, false, MPI_COMM_WORLD);
    d_input = create_d_input_tensor<Tensor>(input);
    output = create_pooling_output_tensor<Tensor>(input, window, strides, pads);
    d_output = create_pooling_d_output_tensor<Tensor>(output);

    if (pid == 0) {
      std::stringstream ss;
      ss << "Input tensor shape: " << input.get_shape()
         << ", distribution: " << input.get_distribution()
         << std::endl
         << "Output tensor shape: " << output.get_shape()
         << ", distribution: " << output.get_distribution()
         << std::endl
         << "Derivative of input tensor shape: "
         << d_input.get_shape() << ", distribution: "
         << d_input.get_distribution()
         << std::endl
         << "Derivative of output tensor shape: "
         << d_output.get_shape() << ", distribution: "
         << d_output.get_distribution() << std::endl;
      std::cout << ss.str();
    }

    if (is_empty()) return;
    // Allocate
    assert0(input.allocate());
    input.zero();
    assert0(output.allocate());
    output.zero();
    assert0(d_input.allocate());
    d_input.zero();
    assert0(d_output.allocate());
    d_output.zero();

    util::MPIPrintStreamDebug() << "Input tensor: " << input
                                << ", output tensor: " << output
                                << ", d_output tensor: " << d_output
                                << ", d_input tensor: " << d_input;
  }
  bool is_empty() const {
    return output.get_size() == 0;
  }
  void initialize() {
    // Initialization
    if (m_cfg.mode == BenchmarkConfig<NSD>::mode_t::SIMPLE) {
      //init_tensor_constant(input, DataType(1.0));
      init_tensor_offset(input, 1);
      //init_tensor_constant(d_output, DataType(1.0));
      init_tensor_offset(d_output, 1);
    } else {
      init_input_tensor(input, input_tensor_seed);
      std::srand(filter_tensor_seed);
      init_input_tensor(d_output, d_output_tensor_seed);
    }
  }
  void dump_input(bool dump_binary) {
    dump_tensor(input, "input_tensor", dump_binary);
    dump_tensor(d_output, "d_output_tensor", dump_binary);
  }
  void dump_output(bool dump_binary) {
    dump_tensor(output, "output_tensor", dump_binary);
    dump_tensor(d_input, "d_input_tensor", dump_binary);
  }
};

template <int NSD, typename Backend, typename DataType>
struct PoolingTester;

template <int NSD, typename DataType>
struct PoolingTester<NSD, ref::Backend, DataType> {
  PoolingTester() {}
  int operator()(Data<NSD, ref::Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg, MPI_Comm comm,
                 Profile<NSD> &prof) {
    util::MPIRootPrintStreamError() << "Not implemented";
    std::abort();
    return 0;
  }
};

template <int NSD, typename Backend, typename DataType>
int test_forward(Data<NSD, Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg,
                 MPI_Comm comm,
                 Backend &be,
                 Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_forward with " << be.get_name();

  Pooling<Backend, 2 + NSD, DataType> pooling(be, cfg.halo_exchange_method);
  pooling.setup(d.input, d.output, d.d_input, d.d_output,
                cfg.f_s, cfg.pads,
                cfg.strides, cfg.pooling_mode);
  MPI_Barrier(comm);
  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    pooling.forward(DataType(1.0), d.input,  DataType(0.0), d.output);
  }
  pooling.wait();
  MPI_Barrier(comm);
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // Run once before starting timing measurement. This
    // will synchronize the processes in a more realistic way
    pooling.forward(DataType(1.0), d.input, DataType(0.0), d.output);
    clk.start();
    pooling.forward(DataType(1.0), d.input, DataType(0.0), d.output);
    clk.stop();
    float elapsed = clk.get_time();
    prof.fwd_time.push_back(elapsed);
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_backward(Data<NSD, Backend, DataType> &d,
                  const BenchmarkConfig<NSD> &cfg,
                  MPI_Comm comm,
                  Backend &be,
                  Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_backward with " << be.get_name();

  Pooling<Backend, 2 + NSD, DataType> pooling(be, cfg.halo_exchange_method);
  pooling.setup(d.input, d.output, d.d_input, d.d_output,
                cfg.f_s,
                cfg.pads,
                cfg.strides, "MAX");

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    pooling.backward(DataType(1.0), d.output, d.d_output,
                     d.input, DataType(0.0), d.d_input);
  }
  pooling.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // Run once before starting timing measurement. This
    // will synchronize the processes in a more realistic way
    pooling.backward(DataType(1.0), d.output, d.d_output,
                     d.input, DataType(0.0), d.d_input);
    clk.start();
    pooling.backward(DataType(1.0), d.output, d.d_output,
                     d.input, DataType(0.0), d.d_input);
    clk.stop();
    float elapsed = clk.get_time();
    prof.bwd_time.push_back(elapsed);
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

#ifdef DISTCONV_HAS_CUDNN
template <int NSD, typename DataType>
struct PoolingTester<NSD, cudnn::BackendCUDNN, DataType> {
  PoolingTester() {}
  int operator()(Data<NSD, cudnn::BackendCUDNN, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg, MPI_Comm comm,
                 Profile<NSD> &prof) {
    int pid;
    DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    cudnnHandle_t cudnn_h;
    DISTCONV_CHECK_CUDNN(cudnnCreate(&cudnn_h));
    cudnn::Options be_opts(cfg.overlap_halo_exchange,
                           cfg.deterministic);
    cudnn::BackendCUDNN be(comm, cudnn_h, be_opts);
    if (cfg.nvtx_marking) {
      be.enable_nvtx_marking();
    }
    test_forward<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, prof);
    test_backward<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, prof);
    return 0;
  }
};
#endif

template <int NSD>
void run(int argc, char *argv[], int pid) {
  auto cfg = process_opt<NSD>(argc, argv, pid, false);
  if (pid == 0) {
    std::cout << cfg << std::endl;
  }

  run_test<NSD, Data, Profile, PoolingTester>(cfg, MPI_COMM_WORLD);

  util::MPIRootPrintStreamInfo() << "Finishing";
}

} // namespace distconv_benchmark

int main(int argc, char *argv[]) {
  distconv_benchmark::set_device();
  int pid;
  DISTCONV_CHECK_MPI(MPI_Init(&argc, &argv));
  DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));

  const int nsd = distconv_benchmark::parse_num_dims(argc, argv);

  if(nsd == 2) {
    distconv_benchmark::run<2>(argc, argv, pid);
  } else if(nsd == 3) {
    distconv_benchmark::run<3>(argc, argv, pid);
  } else {
    util::MPIRootPrintStreamError() << "Invalid --num-dims: " << nsd;
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  DISTCONV_CHECK_MPI(MPI_Finalize());
  return 0;
}
