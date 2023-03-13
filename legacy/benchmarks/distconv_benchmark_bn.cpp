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
#if H2_HAS_ROCM
#include "distconv/util/util_rocm.hpp"
#endif
#ifdef DISTCONV_HAS_CUDNN
#include "distconv/util/util_gpu_dnn.hpp"
#endif

#include <Al.hpp>

namespace distconv_benchmark {

template <int NSD>
class Profile {
 public:
  BenchmarkConfig<NSD> m_cfg;
  std::vector<float> fwd_time;
  std::vector<float> fwd_allreduce_time;
  std::vector<float> bwd_time;
  std::vector<float> bwd_allreduce_time;
  Profile(const BenchmarkConfig<NSD> &cfg):
      m_cfg(cfg),
      fwd_time(cfg.run_count, 0),
      fwd_allreduce_time(cfg.run_count, 0),
      bwd_time(cfg.run_count, 0),
      bwd_allreduce_time(cfg.run_count, 0) {}

  std::ostream &print_as_row(std::ostream &os) {
    for (size_t i = 0; i < fwd_time.size(); ++i) {
      m_cfg.print_as_row(os) << " " << fwd_time[i]
                             << " " << fwd_allreduce_time[i]
                             << " " << bwd_time[i]
                             << " " << bwd_allreduce_time[i];
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
              << "Forward allreduce mean: " << get_mean(fwd_allreduce_time)
              << ", median: " << get_median(fwd_allreduce_time)
              << ", min: " << get_min(fwd_allreduce_time)
              << ", max: " << get_max(fwd_allreduce_time)
              << "\n"
              << "Backward mean: " << get_mean(bwd_time)
              << ", median: " << get_median(bwd_time)
              << ", min: " << get_min(bwd_time)
              << ", max: " << get_max(bwd_time)
              << "\n"
              << "Backward allreduce mean: " << get_mean(bwd_allreduce_time)
              << ", median: " << get_median(bwd_allreduce_time)
              << ", min: " << get_min(bwd_allreduce_time)
              << ", max: " << get_max(bwd_allreduce_time)
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
  typename TensorType<Backend, DataType>::type mean_and_var;
  typename TensorType<Backend, DataType>::type mean;
  typename TensorType<Backend, DataType>::type var;;
  typename TensorType<Backend, DataType>::type running_mean;
  typename TensorType<Backend, DataType>::type running_var;;
  typename TensorType<Backend, DataType>::type scale;
  typename TensorType<Backend, DataType>::type bias;
  typename TensorType<Backend, DataType>::type d_scale;
  typename TensorType<Backend, DataType>::type d_bias;
  typename TensorType<Backend, DataType>::type d_mean_and_var;
  typename TensorType<Backend, DataType>::type d_mean;
  typename TensorType<Backend, DataType>::type d_var;

  Data(const BenchmarkConfig<NSD> &cfg, MPI_Comm comm): m_cfg(cfg) {

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
    tensor::Shape ch_stat_shape2(NSD + 2, 1);
    ch_stat_shape2[-2] = cfg.i_c * 2;
    mean_and_var = Tensor(ch_stat_shape2, loc, shared_dist);
    mean = Tensor(ch_stat_shape, loc, shared_dist);
    var = Tensor(ch_stat_shape, loc, shared_dist);
    running_mean = Tensor(ch_stat_shape, loc, shared_dist);
    running_var = Tensor(ch_stat_shape, loc, shared_dist);
    scale = Tensor(ch_stat_shape, loc, shared_dist);
    bias = Tensor(ch_stat_shape, loc, shared_dist);
    d_scale = Tensor(ch_stat_shape, loc, shared_dist);
    d_bias = Tensor(ch_stat_shape, loc, shared_dist);
    d_mean_and_var = Tensor(ch_stat_shape2, loc, shared_dist);
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
    assert0(mean_and_var.allocate());
    mean_and_var.zero(); // will be overwritten
    assert0(tensor::View(mean, mean_and_var.get_buffer()));
    assert0(tensor::View(var, mean_and_var.get_buffer() +
                         ch_stat_shape.get_size()));
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
    assert0(d_mean_and_var.allocate());
    d_mean_and_var.zero(); // will be overwritten
    assert0(tensor::View(d_mean, d_mean_and_var.get_buffer()));
    assert0(tensor::View(d_var, d_mean_and_var.get_buffer() +
                         ch_stat_shape.get_size()));
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
    if (!m_cfg.skip_weight_allreduce) {
      dump_shared_tensor(d_scale, "d_scale_tensor", dump_binary);
      dump_shared_tensor(d_bias, "d_bias_tensor", dump_binary);
    }
    dump_shared_tensor(d_mean, "d_mean_tensor", dump_binary);
    dump_shared_tensor(d_var, "d_var_tensor", dump_binary);
  }
};

template <int NSD, typename Backend, typename DataType>
int test_forward(Data<NSD, Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg,
                 MPI_Comm comm,
                 Backend &be,
                 BatchNormalization<Backend, DataType> &bn,
                 Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_forward with "
      << be.get_name();

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }

  const bool is_training = true;

  for (int i = 0; i < cfg.warming_up_count; ++i) {
    bn.forward(d.input, d.mean, d.var, d.running_mean, d.running_var,
               d.scale, d.bias, d.output, is_training);
  }

  be.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  DISTCONV_CHECK_MPI(MPI_Barrier(comm));

  Clock<Backend> clk(be);
  Clock<Backend> clk_allreduce(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // Runs for synchronization
    bn.forward_allreduce(d.mean, d.var, is_training);
#ifdef DISTCONV_HAS_NVSHMEM
    if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
      util::nvshmem::launch_barrier(be.get_stream());
    }
#endif // DISTCONV_HAS_NVSHMEM
    // Start measurement
    clk_allreduce.start();
    bn.forward_allreduce(d.mean, d.var, is_training);
    clk_allreduce.stop();
#ifdef DISTCONV_HAS_NVSHMEM
    if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
      util::nvshmem::launch_barrier(be.get_stream());
    }
#endif // DISTCONV_HAS_NVSHMEM
    clk.start();
    bn.forward(d.input, d.mean, d.var, d.running_mean, d.running_var,
               d.scale, d.bias, d.output, is_training);
    clk.stop();
    prof.fwd_time[i] = clk.get_time();
    prof.fwd_allreduce_time[i] = clk_allreduce.get_time();
  }

#ifdef DISTCONV_HAS_NVSHMEM
  if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
    util::nvshmem::barrier();
  }
#endif // DISTCONV_HAS_NVSHMEM
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";
  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_backward(Data<NSD, Backend, DataType> &d,
                  const BenchmarkConfig<NSD> &cfg,
                  MPI_Comm comm,
                  Backend &be,
                  BatchNormalization<Backend, DataType> &bn,
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
                       d.d_scale, d.d_bias, d.d_mean, d.d_var);
    bn.backward_allreduce(d.d_scale, d.d_bias, d.d_mean, d.d_var);
    bn.backward_stage2(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_mean, d.d_var, d.d_input);
  }
  be.wait();
  util::check_for_device_runtime_error();

  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  Clock<Backend> clk_allreduce(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // synchronize the processes
    bn.backward_allreduce(d.d_scale, d.d_bias, d.d_mean, d.d_var);
#ifdef DISTCONV_HAS_NVSHMEM
    if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
      util::nvshmem::launch_barrier(be.get_stream());
    }
#endif // DISTCONV_HAS_NVSHMEM
    clk_allreduce.start();
    bn.backward_allreduce(d.d_scale, d.d_bias, d.d_mean, d.d_var,
                          cfg.skip_weight_allreduce);
    clk_allreduce.stop();
#ifdef DISTCONV_HAS_NVSHMEM
    if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
      util::nvshmem::launch_barrier(be.get_stream());
    }
#endif // DISTCONV_HAS_NVSHMEM
    clk.start();
    bn.backward_stage1(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_scale, d.d_bias, d.d_mean, d.d_var);
    bn.backward_allreduce(d.d_scale, d.d_bias, d.d_mean, d.d_var,
                          cfg.skip_weight_allreduce);
    bn.backward_stage2(d.input, d.d_output, d.mean, d.var, d.scale,
                       d.d_mean, d.d_var, d.d_input);
    clk.stop();
    prof.bwd_time[i] = clk.get_time();
    prof.bwd_allreduce_time[i] = clk_allreduce.get_time();
  }

#ifdef DISTCONV_HAS_NVSHMEM
  if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
    util::nvshmem::barrier();
  }
#endif // DISTCONV_HAS_NVSHMEM
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
struct BNTester<NSD, BackendDNNLib, DataType> {
  BNTester() {}
  int operator()(Data<NSD, BackendDNNLib, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg, MPI_Comm comm,
                 Profile<NSD> &prof) {
    int pid;
    DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
    cudnnHandle_t cudnn_h;
    DISTCONV_CHECK_CUDNN(cudnnCreate(&cudnn_h));
    BackendOptions be_opts(cfg.overlap_halo_exchange,
                           cfg.deterministic,
                           cfg.profiling);
    BackendDNNLib be(comm, cudnn_h, be_opts);
    BatchNormalization<BackendDNNLib, DataType> bn(
        be, 2 + NSD, 0.9, 1e-5, cfg.global_stat, cfg.batchnorm_impl);
    bn.set_num_samples(d.input.get_shape()[-1]);
    start_profiler<BackendDNNLib>();
    if (cfg.nvtx_marking) {
      be.enable_nvtx_marking();
    }

#ifdef DISTCONV_HAS_NVSHMEM
    // hack to use NVSHMEM buffers
    if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
      // mean and var
      auto mean_and_var_nvshmem = static_cast<DataType*>(
          nvshmem_malloc(d.mean_and_var.get_max_local_real_shape().size()
                         * sizeof(DataType)));
      assert_always(mean_and_var_nvshmem != nullptr);
      assert0(tensor::View(d.mean_and_var, mean_and_var_nvshmem));
      assert0(tensor::View(d.mean, d.mean_and_var.get_buffer()));
      assert0(tensor::View(d.var, d.mean_and_var.get_buffer() +
                           d.mean.get_local_real_size()));
      // d_mean and d_var
      auto d_mean_and_var_nvshmem = static_cast<DataType*>(
          nvshmem_malloc(d.d_mean_and_var.get_max_local_real_shape().size()
                         * sizeof(DataType)));
      assert_always(d_mean_and_var_nvshmem != nullptr);
      assert0(tensor::View(d.d_mean_and_var, d_mean_and_var_nvshmem));
      assert0(tensor::View(d.d_mean, d.d_mean_and_var.get_buffer()));
      assert0(tensor::View(d.d_var, d.d_mean_and_var.get_buffer() +
                           d.d_mean.get_local_real_size()));
      // scale_gradient
      auto d_scale_nvshmem = static_cast<DataType*>(
          nvshmem_malloc(d.d_scale.get_max_local_real_shape().size()
                         * sizeof(DataType)));
      assert_always(d_scale_nvshmem != nullptr);
      assert0(tensor::View(d.d_scale, d_scale_nvshmem));
      // bias_gradient
      auto d_bias_nvshmem = static_cast<DataType*>(
          nvshmem_malloc(d.d_bias.get_max_local_real_shape().size()
                         * sizeof(DataType)));
      assert_always(d_bias_nvshmem != nullptr);
      assert0(tensor::View(d.d_bias, d_bias_nvshmem));
    }
#endif // DISTCONV_HAS_NVSHMEM

    test_forward<NSD, BackendDNNLib, DataType>(
        d, cfg, comm, be, bn, prof);
    test_backward<NSD, BackendDNNLib, DataType>(
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
  if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
    util::nvshmem::initialize(MPI_COMM_WORLD);
  }
#endif // DISTCONV_HAS_NVSHMEM

  run_test<NSD, Data, Profile, BNTester>(cfg, MPI_COMM_WORLD);

  util::MPIRootPrintStreamInfo() << "Finishing";

#ifdef DISTCONV_HAS_NVSHMEM
  if (IsNVSHMEMUsed(cfg.batchnorm_impl)) {
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
