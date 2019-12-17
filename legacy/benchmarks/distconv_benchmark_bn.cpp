#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cstdlib>
#include <limits>

#include "distconv_config.hpp"
#include "distconv_benchmark_common.hpp"
#include "benchmark_common.hpp"

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

#include <Al.hpp>

namespace distconv_benchmark {

template <int NSD>
class Profile {
 public:
  BenchmarkConfig<NSD> m_cfg;
  std::vector<float> fwd_time;
  std::vector<float> bwd_time;
  Profile(const BenchmarkConfig<NSD> &cfg):
      m_cfg(cfg),
      fwd_time(cfg.run_count, 0),
      bwd_time(cfg.run_count, 0) {}

  std::ostream &print_as_row(std::ostream &os) {
    for (size_t i = 0; i < fwd_time.size(); ++i) {
      m_cfg.print_as_row(os) << " " << fwd_time[i]
                             << " " << bwd_time[i];
      os << std::endl;
    }
    return os;
  }

  void print_summary(std::ostream &os) {
    std::cout << "Forward mean: " << get_mean(fwd_time)
              << ", median: " << get_median(fwd_time)
              << ", min: " << get_min(fwd_time)
              << ", max: " << get_max(fwd_time)
              << "\n"
              << "Backward mean: "
              << get_mean(bwd_time)
              << ", median: " << get_median(bwd_time)
              << ", min: " << get_min(bwd_time)
              << ", max: " << get_max(bwd_time)
              << std::endl;
  }
};

template <int NSD, typename Backend, typename DataType>
class Data {
 public:
  const BenchmarkConfig<NSD> &m_cfg;
  using Tensor = typename TensorType<Backend, DataType>::type;
  typename TensorType<Backend, DataType>::type input;
  typename TensorType<Backend, DataType>::type d_input;
  typename TensorType<Backend, DataType>::type output;
  typename TensorType<Backend, DataType>::type d_output;
  typename TensorType<Backend, DataType>::type mean;
  typename TensorType<Backend, DataType>::type var;;
  typename TensorType<Backend, DataType>::type running_mean;
  typename TensorType<Backend, DataType>::type running_var;;
  typename TensorType<Backend, DataType>::type scale;
  typename TensorType<Backend, DataType>::type bias;
  typename TensorType<Backend, DataType>::type d_scale;
  typename TensorType<Backend, DataType>::type d_bias;
  typename TensorType<Backend, DataType>::type d_mean;
  typename TensorType<Backend, DataType>::type d_var;

  Data(const BenchmarkConfig<NSD> &cfg, MPI_Comm comm): m_cfg(cfg) {
    using Tensor = typename TensorType<Backend, DataType>::type;

    int pid;
    int np;
    MPI_Comm_rank(comm, &pid);
    MPI_Comm_size(comm, &np);

    assert_eq(std::accumulate(cfg.p_s.begin(),
                              cfg.p_s.end(),
                              1,
                              std::multiplies<int>())
              * cfg.p_c * cfg.p_n, np);

    const auto vector_concat = [](const int_vector v, const int c, const int n) {
                                 int_vector cn({c, n});
                                 cn.insert(cn.begin(), v.begin(), v.end());
                                 return (const int_vector) cn;
                               };

    const auto input_shape  = vector_concat(cfg.i_s, cfg.i_c, cfg.i_n);
    const auto locale_shape = vector_concat(cfg.p_s, cfg.p_c, cfg.p_n);
    util::MPIPrintStreamDebug()
      << "input_shape: " << util::join_array(input_shape, " ")
      << " locale_shape: " << util::join_array(locale_shape, " ");

    const auto dist = tensor::Distribution::make_distribution(
        tensor::Shape(locale_shape));
    // This assumes no partitioning of the channel dimension
    const auto shared_dist = tensor::Distribution::make_shared_distribution(
        tensor::Shape(locale_shape));
    const auto loc = tensor::LocaleMPI(comm);

    input = Tensor(tensor::Shape(input_shape), loc, dist);
    d_input = Tensor(tensor::Shape(input_shape), loc, dist);
    output = Tensor(tensor::Shape(input_shape), loc, dist);
    d_output = Tensor(tensor::Shape(input_shape), loc, dist);

    tensor::Shape ch_stat_shape(NSD + 2, 1);
    ch_stat_shape[-2] = cfg.i_c;
    mean = Tensor(ch_stat_shape, loc, shared_dist);
    var = Tensor(ch_stat_shape, loc, shared_dist);
    running_mean = Tensor(ch_stat_shape, loc, shared_dist);
    running_var = Tensor(ch_stat_shape, loc, shared_dist);
    scale = Tensor(ch_stat_shape, loc, shared_dist);
    bias = Tensor(ch_stat_shape, loc, shared_dist);
    d_scale = Tensor(ch_stat_shape, loc, shared_dist);
    d_bias = Tensor(ch_stat_shape, loc, shared_dist);
    d_mean = Tensor(ch_stat_shape, loc, shared_dist);
    d_var = Tensor(ch_stat_shape, loc, shared_dist);

    if (pid == 0) {
      std::cout << "Tensor shape: " << input.get_shape()
                << ", distribution: " << input.get_distribution() << "\n";
    }

    // If the output is empty, skip allocation.
    if (is_empty()) return;
    // Allocate
    assert0(input.allocate());
    input.zero(); // will be used
    assert0(output.allocate());
    output.zero(); // will be overwritten
    assert0(d_input.allocate());
    d_input.zero(); // will be overwritten
    assert0(d_output.allocate());
    d_output.zero(); // will be used
    assert0(mean.allocate());
    mean.zero(); // will be overwritten
    assert0(var.allocate());
    var.zero(); // will be overwritten
    assert0(running_mean.allocate());
    running_mean.zero(); // will be incremented
    assert0(running_var.allocate());
    running_var.zero(); // will be incremented
    assert0(scale.allocate());
    scale.zero(); // will be used
    assert0(bias.allocate());
    bias.zero(); // will be used
    assert0(d_scale.allocate());
    d_scale.zero(); // will be overwritten
    assert0(d_bias.allocate());
    d_bias.zero(); // will be overwritten
    assert0(d_mean.allocate());
    d_mean.zero(); // will be overwritten
    assert0(d_var.allocate());
    d_var.zero(); // will be overwritten
  }
  bool is_empty() const {
    return output.get_size() == 0;
  }
  void initialize() {
    // Initialization
    if (m_cfg.mode == BenchmarkConfig<NSD>::mode_t::SIMPLE) {
      init_tensor_constant(input, DataType(1.0));
      init_tensor_constant(d_output, DataType(1.0));
    } else {
      init_input_tensor(input, input_tensor_seed);
      init_input_tensor(d_output, d_output_tensor_seed);
    }
    init_tensor_constant(scale, 1);
    init_tensor_constant(bias, 0);
    init_tensor_constant(running_mean, 0);
    init_tensor_constant(running_var, 1);
  }
  void dump_input(bool dump_binary) {
    dump_tensor(input, "input_tensor", dump_binary);
    dump_tensor(d_output, "d_output_tensor", dump_binary);
  }
  void dump_output(bool dump_binary) {
    dump_tensor(output, "output_tensor", dump_binary);
    dump_tensor(d_input, "d_input_tensor", dump_binary);
    dump_shared_tensor(mean, "mean_tensor", dump_binary);
    dump_shared_tensor(var, "var_tensor", dump_binary);
    dump_shared_tensor(running_mean, "running_mean_tensor", dump_binary);
    dump_shared_tensor(running_var, "running_var_tensor", dump_binary);
    dump_shared_tensor(d_scale, "d_scale_tensor", dump_binary);
    dump_shared_tensor(d_bias, "d_bias_tensor", dump_binary);
    dump_shared_tensor(d_mean, "d_mean_tensor", dump_binary);
    dump_shared_tensor(d_var, "d_var_tensor", dump_binary);
  }
};

template <int NSD, typename Backend, typename DataType>
int test_forward(Data<NSD, Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg,
                 MPI_Comm comm,
                 Backend &be,
                 BatchNormalization<Backend, NSD+2, DataType> &bn,
                 Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_forward with "
      << be.get_name();

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }

  for (int i = 0; i < cfg.warming_up_count; ++i) {
    bn.forward(d.input, d.mean, d.var, d.running_mean, d.running_var,
               d.scale, d.bias, d.output, true);
  }

  be.wait();

  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // Runs for synchronization
    bn.forward(d.input, d.mean, d.var, d.running_mean, d.running_var,
               d.scale, d.bias, d.output, true);
    // Start measurement
    clk.start();
    bn.forward(d.input, d.mean, d.var, d.running_mean, d.running_var,
               d.scale, d.bias, d.output, true);
    clk.stop();
    float elapsed = clk.get_time();
    prof.fwd_time[i] = elapsed;
  }

  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done\n";
  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_backward(Data<NSD, Backend, DataType> &d,
                  const BenchmarkConfig<NSD> &cfg,
                  MPI_Comm comm,
                  Backend &be,
                  BatchNormalization<Backend, NSD+2, DataType> &bn,
                  Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_backward with "
      << be.get_name();

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }

  for (int i = 0; i < cfg.warming_up_count; ++i) {
    bn.backward_stage1(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_scale, d.d_bias, d.d_mean, d.d_var,
                       cfg.global_stat);
    bn.backward_stage2(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_mean, d.d_var, d.d_input);
  }
  be.wait();
  DISTCONV_CHECK_CUDA(cudaGetLastError());

  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  Clock<Backend> clk_data(be);
  Clock<Backend> clk_filter(be);
  Clock<Backend> clk_bias(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // synchronize the processes
    bn.backward_stage1(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_scale, d.d_bias, d.d_mean, d.d_var,
                       cfg.global_stat);
    clk.start();
    bn.backward_stage1(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_scale, d.d_bias, d.d_mean, d.d_var,
                       cfg.global_stat);
    bn.backward_stage2(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_mean, d.d_var, d.d_input);
    clk.stop();
    float elapsed = clk.get_time();
    prof.bwd_time[i] = elapsed;
  }

  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";
  return 0;
}

template <int NSD, typename Backend, typename DataType>
struct BNTester;

// Not implemented yet
template <int NSD, typename DataType>
struct BNTester<NSD, ref::Backend, DataType> {
  BNTester() {}
  int operator()(Data<NSD, ref::Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg, MPI_Comm comm,
                 Profile<NSD> &prof) {
    return 0;
  }
};

#ifdef DISTCONV_HAS_CUDNN
template <int NSD, typename DataType>
struct BNTester<NSD, cudnn::BackendCUDNN, DataType> {
  BNTester() {}
  int operator()(Data<NSD, cudnn::BackendCUDNN, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg, MPI_Comm comm,
                 Profile<NSD> &prof) {
    int pid;
    DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    cudnnHandle_t cudnn_h;
    DISTCONV_CHECK_CUDNN(cudnnCreate(&cudnn_h));
    cudnn::Options be_opts(cfg.overlap_halo_exchange,
                           cfg.deterministic,
                           cfg.profiling);
    cudnn::BackendCUDNN be(comm, cudnn_h, be_opts);
    BatchNormalization<cudnn::BackendCUDNN, 2 + NSD, DataType> bn(
        be, 0.9, 1e-5, std::vector<bool>(2 + NSD, cfg.global_stat),
        cfg.batchnorm_impl);
    bn.set_num_samples(d.input.get_shape()[-1]);
    start_profiler<cudnn::BackendCUDNN>();
    if (cfg.nvtx_marking) {
      be.enable_nvtx_marking();
    }
    test_forward<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, bn, prof);
    test_backward<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, bn, prof);
    return 0;
  }
};
#endif

template <int NSD>
void run(int argc, char *argv[], int pid, int np) {
  auto cfg = process_opt<NSD>(argc, argv, pid, true);
  if (pid == 0) {
    std::cout << cfg << std::endl;
  }

  if (std::accumulate(cfg.p_s.begin(),
                      cfg.p_s.end(),
                      1,
                      std::multiplies<int>())
      * cfg.p_c * cfg.p_n != np) {
    util::MPIRootPrintStreamError()
        << "Number of ranks does not match with the number of tensor partitions";
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

#ifdef DISTCONV_HAS_NVSHMEM
  if (IsNVSHMEMUsed(cfg.halo_exchange_method)) {
    util::nvshmem::initialize(MPI_COMM_WORLD);
  }
#endif // DISTCONV_HAS_NVSHMEM

  run_test<NSD, Data, Profile, BNTester>(cfg, MPI_COMM_WORLD);

  util::MPIRootPrintStreamInfo() << "Finishing";

#ifdef DISTCONV_HAS_NVSHMEM
  if (IsNVSHMEMUsed(cfg.halo_exchange_method)) {
    util::nvshmem::finalize();
  }
#endif // DISTCONV_HAS_NVSHMEM
}

} // namespace distconv_benchmark

int main(int argc, char *argv[]) {
  distconv_benchmark::set_device();
  int pid;
  int np;
  Al::Initialize(argc, argv);
  DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  DISTCONV_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &np));

  const int nsd = distconv_benchmark::parse_num_dims(argc, argv);

  if(nsd == 2) {
    distconv_benchmark::run<2>(argc, argv, pid, np);
  } else if(nsd == 3) {
    distconv_benchmark::run<3>(argc, argv, pid, np);
  } else {
    distconv::util::MPIRootPrintStreamError() << "Invalid --num-dims: " << nsd;
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  Al::Finalize();
  return 0;
}
