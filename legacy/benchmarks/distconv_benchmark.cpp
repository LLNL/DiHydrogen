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

using namespace distconv;

namespace distconv_benchmark {

template <int NSD>
class Profile {
 public:
  BenchmarkConfig<NSD> m_cfg;
  std::vector<float> conv_fwd_time;
  std::vector<float> conv_bwd_data_time;
  std::vector<float> conv_bwd_filter_time;
  std::vector<float> conv_bwd_bias_time;
  std::vector<float> conv_bwd_combined_data_time;
  std::vector<float> conv_bwd_combined_filter_time;
  std::vector<float> conv_bwd_combined_bias_time;
  std::vector<float> conv_bwd_combined_all_time;
  Profile(const BenchmarkConfig<NSD> &cfg):
      m_cfg(cfg),
      conv_fwd_time(cfg.run_count, 0),
      conv_bwd_data_time(cfg.run_count, 0),
      conv_bwd_filter_time(cfg.run_count, 0),
      conv_bwd_bias_time(cfg.run_count, 0),
      conv_bwd_combined_data_time(cfg.run_count, 0),
      conv_bwd_combined_filter_time(cfg.run_count, 0),
      conv_bwd_combined_bias_time(cfg.run_count, 0),
      conv_bwd_combined_all_time(cfg.run_count, 0) {}

  std::ostream &print_as_row(std::ostream &os) {
    for (size_t i = 0; i < conv_fwd_time.size(); ++i) {
      m_cfg.print_as_row(os) << " " << conv_fwd_time[i]
                             << " " << conv_bwd_data_time[i]
                             << " " << conv_bwd_filter_time[i]
                             << " " << conv_bwd_bias_time[i];
      os << " " << conv_bwd_combined_data_time[i]
         << " " << conv_bwd_combined_filter_time[i]
         << " " << conv_bwd_combined_bias_time[i]
         << " " << conv_bwd_combined_all_time[i];
      os << std::endl;
    }
    return os;
  }

  void print_summary(std::ostream &os) {
    std::cout << "Forward mean: " << get_mean(conv_fwd_time)
              << ", median: " << get_median(conv_fwd_time)
              << ", min: " << get_min(conv_fwd_time)
              << ", max: " << get_max(conv_fwd_time)
              << "\n";
#if 0
    std::cout << "Backward data mean: " << get_mean(conv_bwd_data_time)
              << ", median: " << get_median(conv_bwd_data_time)
              << ", min: " << get_min(conv_bwd_data_time)
              << ", max: " << get_max(conv_bwd_data_time)
              << "\n";
    std::cout << "Backward filter mean: " << get_mean(conv_bwd_filter_time)
              << ", median: " << get_median(conv_bwd_filter_time)
              << ", min: " << get_min(conv_bwd_filter_time)
              << ", max: " << get_max(conv_bwd_filter_time)
              << "\n";
    if (m_cfg.use_bias) {
      std::cout << "Backward bias mean: " << get_mean(conv_bwd_bias_time)
                << ", median: " << get_median(conv_bwd_bias_time)
                << ", min: " << get_min(conv_bwd_bias_time)
                << ", max: " << get_max(conv_bwd_bias_time)
                << "\n";
    }
#endif
    std::cout << "Backward combined data mean: "
              << get_mean(conv_bwd_combined_data_time)
              << ", median: " << get_median(conv_bwd_combined_data_time)
              << ", min: " << get_min(conv_bwd_combined_data_time)
              << ", max: " << get_max(conv_bwd_combined_data_time)
              << "\n";
    std::cout << "Backward combined filter mean: "
              << get_mean(conv_bwd_combined_filter_time)
              << ", median: " << get_median(conv_bwd_combined_filter_time)
              << ", min: " << get_min(conv_bwd_combined_filter_time)
              << ", max: " << get_max(conv_bwd_combined_filter_time)
              << "\n";
    if (m_cfg.use_bias) {
      std::cout << "Backward combined bias mean: "
                << get_mean(conv_bwd_combined_bias_time)
                << ", median: " << get_median(conv_bwd_combined_bias_time)
                << ", min: " << get_min(conv_bwd_combined_bias_time)
                << ", max: " << get_max(conv_bwd_combined_bias_time)
                << "\n";
    }
    std::cout << "Backward combined all mean: "
              << get_mean(conv_bwd_combined_all_time)
              << ", median: " << get_median(conv_bwd_combined_all_time)
              << ", min: " << get_min(conv_bwd_combined_all_time)
              << ", max: " << get_max(conv_bwd_combined_all_time)
              << "\n";
  }
};

template <int NSD, typename Backend, typename DataType>
class Data {
 public:
  const BenchmarkConfig<NSD> &m_cfg;
  typename TensorType<Backend, DataType>::type input;
  typename TensorType<Backend, DataType>::type output;
  typename TensorType<Backend, DataType>::type filter;
  typename TensorType<Backend, DataType>::type bias;
  typename TensorType<Backend, DataType>::type d_input;
  typename TensorType<Backend, DataType>::type d_output;
  typename TensorType<Backend, DataType>::type d_filter;
  typename TensorType<Backend, DataType>::type d_bias;

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

    const auto vector_concat = [](const int_vector v, const int c, const int n) {
                                 int_vector cn({c, n});
                                 cn.insert(cn.begin(), v.begin(), v.end());
                                 return (const int_vector) cn;
                               };

    const auto input_shape  = vector_concat(cfg.i_s, cfg.i_c, cfg.i_n);
    const auto locale_shape = vector_concat(cfg.p_s, cfg.p_c, cfg.p_n);
    const auto filter_dims = cfg.f_s;
    const auto strides = cfg.strides;
    const auto pads = cfg.pads;
    const auto dilations = cfg.dilations;
    util::MPIPrintStreamDebug()
      << "input_shape: " << util::join_array(input_shape, " ")
      << " locale_shape: " << util::join_array(locale_shape, " ");
    input = create_input_tensor<Tensor>(
        input_shape, locale_shape, filter_dims, strides,
        dilations, cfg.deconv, MPI_COMM_WORLD);
    d_input = create_d_input_tensor<Tensor>(input);

    filter = create_filter_tensor<Tensor>(locale_shape, filter_dims, input,
                                          cfg.i_c, cfg.f_k, cfg.num_groups,
                                          MPI_COMM_WORLD,
                                          cfg.chanfilt_algo, cfg.p_f);
    d_filter = create_d_filter_tensor<Tensor>(filter);

    if (!cfg.deconv) {
      output = create_convolution_output_tensor<Tensor>(input, filter,
                                                        strides, pads, dilations,
                                                        cfg.num_groups);
      d_output = create_convolution_d_output_tensor<Tensor>(
          output, filter, dilations);
    } else {
      output = create_deconvolution_output_tensor<Tensor>(input, filter,
                                                          strides, pads, dilations,
                                                          cfg.num_groups);
      d_output = create_deconvolution_d_output_tensor<Tensor>(
          output, filter, dilations);
    }

    if (cfg.use_bias) {
      bias = create_bias_tensor<Tensor>(output);
      d_bias = create_d_bias_tensor<Tensor>(output);
    }

    if (pid == 0) {
      std::cout << "Input tensor shape: " << input.get_shape()
                << ", distribution: " << input.get_distribution() << "\n";
      std::cout << "Filter tensor shape: " << filter.get_shape()
                << ", distribution: " << filter.get_distribution() << "\n";
      std::cout << "Output tensor shape: " << output.get_shape()
                << ", distribution: " << output.get_distribution() << "\n";
      //std::cout << "Derivative of input tensor distribution: "
      //<< d_input_dist << "\n";

      std::cout << "Derivative of output tensor distribution: "
                << d_output.get_distribution() << "\n";
      if (cfg.use_bias) {
        std::cout << "Bias tensor shape: " << bias.get_shape()
                  << ", distribution: " << bias.get_distribution() << "\n";
      }
    }

    // If the output is empty, skip allocation.
    if (is_empty()) return;
    // Allocate
    assert0(input.allocate());
    input.zero();
    assert0(filter.allocate());
    filter.zero();
    assert0(output.allocate());
    output.zero();
    assert0(d_input.allocate());
    d_input.zero();
    assert0(d_filter.allocate());
    d_filter.zero();
    assert0(d_output.allocate());
    d_output.zero();
    if (cfg.use_bias) {
      assert0(bias.allocate());
      bias.zero();
      assert0(d_bias.allocate());
      d_bias.zero();
    }
  }
  bool is_empty() const {
    return output.get_size() == 0;
  }
  void initialize() {
    // Initialization
    if (m_cfg.mode == BenchmarkConfig<NSD>::mode_t::SIMPLE) {
      init_tensor_constant(input, DataType(1.0));
      init_tensor_offset(filter);
      init_tensor_constant(d_output, DataType(1.0));
    } else {
      init_input_tensor(input, input_tensor_seed);
      init_input_tensor(filter, filter_tensor_seed);
      init_input_tensor(d_output, d_output_tensor_seed);
    }
    if (m_cfg.use_bias) {
      init_tensor_constant(bias, 0.01);
    }
  }
  void dump_input(bool dump_binary) {
    dump_tensor(input, "input_tensor", dump_binary);
    dump_tensor(filter, "filter_tensor", dump_binary);
    dump_tensor(d_output, "d_output_tensor", dump_binary);
    dump_shared_tensor(bias, "bias_tensor", dump_binary);
  }
  void dump_output(bool dump_binary) {
    dump_tensor(output, "output_tensor", dump_binary);
    dump_tensor(d_input, "d_input_tensor", dump_binary);
    dump_tensor(d_filter, "d_filter_tensor", dump_binary);
    dump_shared_tensor(d_bias, "d_bias_tensor", dump_binary);
  }
};

template <int NSD, typename Backend, typename DataType>
int test_convolution_forward(Data<NSD, Backend, DataType> &d,
                             const BenchmarkConfig<NSD> &cfg,
                             MPI_Comm comm,
                             Backend &be,
                             Convolution<Backend, DataType> &conv,
                             Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_convolution_forward with "
      << be.get_name();

  if (cfg.skip_halo_exchange) {
    util::MPIRootPrintStreamInfo() << "Skipping halo exchange";
  }

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    conv.forward(DataType(1.0), d.input, d.filter, DataType(0.0),
                 d.output, cfg.skip_halo_exchange,
                 cfg.skip_chanfilt_comm);
    if (cfg.use_bias) {
      conv.apply_bias(DataType(1.0), d.bias, DataType(1.0), d.output);
    }
  }
  conv.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    // Run convolution before starting timing measurement. This
    // will synchronize the processes in a more realistic way
    for (int j = 0;
         j < std::max(2, (*std::max_element(cfg.p_s.begin(), cfg.p_s.end())) - 1);
         ++j) {
      conv.forward(DataType(1.0), d.input, d.filter, DataType(0.0),
                   d.output, cfg.skip_halo_exchange, cfg.skip_chanfilt_comm,
                   false);
    }
    clk.start();
    conv.forward(DataType(1.0), d.input, d.filter, DataType(0.0),
                 d.output, cfg.skip_halo_exchange, cfg.skip_chanfilt_comm,
                 false);
    if (cfg.use_bias) {
      conv.apply_bias(DataType(1.0), d.bias, DataType(1.0), d.output);
    }
    clk.stop();
    float elapsed = clk.get_time();
    prof.conv_fwd_time[i] = elapsed;
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done\n";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_convolution_backward_data(Data<NSD, Backend, DataType> &d,
                                   const BenchmarkConfig<NSD> &cfg,
                                   MPI_Comm comm,
                                   Backend &be,
                                   Convolution<Backend, DataType> &conv,
                                   Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_convolution_backward_data with "
      << be.get_name();

  if (cfg.skip_halo_exchange) {
    util::MPIRootPrintStreamInfo() << "Skipping halo exchange";
  }

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }

  bool call_halo_exch_separately =
      conv.is_overlap_bwd_halo_exchange_enabled() &&
      !cfg.skip_halo_exchange;

  for (int i = 0; i < cfg.warming_up_count; ++i) {
    if (call_halo_exch_separately) {
      conv.backward_data_exchange_halo(d.d_output);
    }
    conv.backward_data(DataType(1.0), d.filter, d.d_output,
                       DataType(0.0), d.d_input,
                       cfg.skip_halo_exchange, cfg.skip_chanfilt_comm);
  }
  conv.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    if (call_halo_exch_separately) {
      conv.backward_data_exchange_halo(d.d_output);
    }
    conv.backward_data(DataType(1.0), d.filter, d.d_output,
                       DataType(0.0), d.d_input,
                       cfg.skip_halo_exchange, cfg.skip_chanfilt_comm, false);
    clk.start();
    if (call_halo_exch_separately) {
      conv.backward_data_exchange_halo(d.d_output);
    }
    conv.backward_data(DataType(1.0), d.filter, d.d_output,
                       DataType(0.0), d.d_input,
                       cfg.skip_halo_exchange, cfg.skip_chanfilt_comm, false);
    clk.stop();
    float elapsed = clk.get_time();
    prof.conv_bwd_data_time[i] = elapsed;
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));

  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_convolution_backward_filter(Data<NSD, Backend, DataType> &d,
                                     const BenchmarkConfig<NSD> &cfg,
                                     MPI_Comm comm,
                                     Backend &be,
                                     Convolution<Backend, DataType> &conv,
                                     Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_convolution_backward_filter with "
      << be.get_name();

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    conv.backward_filter(DataType(1.0), d.input, d.d_output,
                         DataType(0.0), d.d_filter, !cfg.skip_weight_allreduce,
                         cfg.skip_chanfilt_comm);
  }
  conv.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    clk.start();
    conv.backward_filter(DataType(1.0), d.input, d.d_output,
                         DataType(0.0), d.d_filter,
                         !cfg.skip_weight_allreduce, cfg.skip_chanfilt_comm, false);
    clk.stop();
    float elapsed = clk.get_time();
    prof.conv_bwd_filter_time[i] = elapsed;
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_convolution_backward_bias(Data<NSD, Backend, DataType> &d,
                                   const BenchmarkConfig<NSD> &cfg,
                                   MPI_Comm comm,
                                   Backend &be,
                                   Convolution<Backend, DataType> &conv,
                                   Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_convolution_backward_bias with "
      << be.get_name();

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    conv.backward_bias(DataType(1.0), d.d_output, DataType(0.0), d.d_bias);
  }
  conv.wait();
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurement";

  Clock<Backend> clk(be);
  for (int i = 0; i < cfg.run_count; ++i) {
    complete_async<Backend>();
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    if (!cfg.testing) {
      spin_async_device(cfg.spin_time_ms, be);
    }
    clk.start();
    conv.backward_bias(DataType(1.0), d.d_output, DataType(0.0),
                       d.d_bias, !cfg.skip_weight_allreduce);
    clk.stop();
    float elapsed = clk.get_time();
    prof.conv_bwd_bias_time[i] = elapsed;
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
int test_convolution_backward(Data<NSD, Backend, DataType> &d,
                              const BenchmarkConfig<NSD> &cfg,
                              MPI_Comm comm,
                              Backend &be,
                              Convolution<Backend, DataType> &conv,
                              Profile<NSD> &prof) {
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
      << "Executing test_convolution_backward with "
      << be.get_name();

  if (cfg.skip_halo_exchange) {
    util::MPIRootPrintStreamInfo() << "Skipping halo exchange";
  }

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }

  bool call_halo_exch_separately =
      conv.is_overlap_bwd_halo_exchange_enabled() &&
      !cfg.skip_halo_exchange;

  for (int i = 0; i < cfg.warming_up_count; ++i) {
    conv.backward_filter(DataType(1.0), d.input, d.d_output,
                         DataType(0.0), d.d_filter,
                         !cfg.skip_weight_allreduce, cfg.skip_chanfilt_comm);
    if (call_halo_exch_separately) {
      conv.backward_data_exchange_halo(d.d_output);
    }
    conv.backward_data(DataType(1.0), d.filter, d.d_output,
                       DataType(0.0), d.d_input, cfg.skip_halo_exchange,
                       cfg.skip_chanfilt_comm);
    if (cfg.use_bias) {
      conv.backward_bias(DataType(1.0), d.d_output, DataType(0.0),
                         d.d_bias, !cfg.skip_weight_allreduce);
    }
  }
  conv.wait();
  util::check_for_device_runtime_error();

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
    if (!cfg.testing) {
      for (int j = 0;
           j < std::max(2, (*std::max_element(cfg.p_s.begin(), cfg.p_s.end())) - 1);
           ++j) {
        if (call_halo_exch_separately) {
          conv.backward_data_exchange_halo(d.d_output);
        }
        conv.backward_filter(DataType(1.0), d.input, d.d_output,
                             DataType(0.0), d.d_filter,
                             !cfg.skip_weight_allreduce, cfg.skip_chanfilt_comm);
        conv.backward_data(DataType(1.0), d.filter, d.d_output,
                           DataType(0.0), d.d_input, cfg.skip_halo_exchange,
                           cfg.skip_chanfilt_comm);
      }
    }
    clk.start();
    if (call_halo_exch_separately) {
      conv.backward_data_exchange_halo(d.d_output);
    }
    clk_filter.start();
    conv.backward_filter(DataType(1.0), d.input, d.d_output,
                         DataType(0.0), d.d_filter,
                         !cfg.skip_weight_allreduce, cfg.skip_chanfilt_comm);
    clk_filter.stop();
    if (cfg.use_bias) {
      clk_bias.start();
      conv.backward_bias(DataType(1.0), d.d_output, DataType(0.0),
                         d.d_bias, !cfg.skip_weight_allreduce);
      clk_bias.stop();
    }
    clk_data.start();
    conv.backward_data(DataType(1.0), d.filter, d.d_output,
                       DataType(0.0), d.d_input,
                       cfg.skip_halo_exchange, cfg.skip_chanfilt_comm);
    clk_data.stop();
    clk.stop();
    if (!cfg.testing) {
      for (int j = 0;
           j < std::max(2, (*std::max_element(cfg.p_s.begin(), cfg.p_s.end())) - 1);
           ++j) {
        if (call_halo_exch_separately) {
          conv.backward_data_exchange_halo(d.d_output);
        }
        conv.backward_filter(DataType(1.0), d.input, d.d_output,
                             DataType(0.0), d.d_filter,
                             !cfg.skip_weight_allreduce, cfg.skip_chanfilt_comm);
        conv.backward_data(DataType(1.0), d.filter, d.d_output,
                           DataType(0.0), d.d_input, cfg.skip_halo_exchange,
                           cfg.skip_chanfilt_comm);
      }
    }
    prof.conv_bwd_combined_filter_time[i] = clk_filter.get_time();
    prof.conv_bwd_combined_data_time[i] = clk_data.get_time();
    if (cfg.use_bias) {
      prof.conv_bwd_combined_bias_time[i] = clk_bias.get_time();
    }
    prof.conv_bwd_combined_all_time[i] = clk.get_time();
  }
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));

  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD, typename Backend, typename DataType>
struct ConvolutionTester;

template <int NSD, typename DataType>
struct ConvolutionTester<NSD, ref::Backend, DataType> {
  ConvolutionTester() {}
  int operator()(Data<NSD, ref::Backend, DataType> &d,
                 const BenchmarkConfig<NSD> &cfg,
                 MPI_Comm comm,
                 Profile<NSD> &prof) {
    ref::Backend be;
    Convolution<ref::Backend, DataType> conv(
        be, 2 + NSD, cfg.halo_exchange_method);
    conv.setup(d.input, d.filter, d.output,
               d.d_input, d.d_filter, d.d_output,
               cfg.pads,
               cfg.strides,
               cfg.dilations,
               cfg.num_groups,
               cfg.conv_fwd_algo, cfg.conv_bwd_data_algo,
               cfg.conv_bwd_filter_algo, 0);
    if (cfg.use_bias) {
      conv.setup_bias(d.bias);
      conv.setup_bias_gradient(d.d_bias);
    }
    d.initialize();

    test_convolution_forward<NSD, ref::Backend, DataType>(
        d, cfg, comm, be, conv, prof);
    test_convolution_backward_data<NSD, ref::Backend, DataType>(
        d, cfg, comm, be, conv, prof);
    test_convolution_backward_filter<NSD, ref::Backend, DataType>(
        d, cfg, comm, be, conv, prof);
    return 0;
  }
};

#ifdef DISTCONV_HAS_CUDNN
template <int NSD, typename DataType>
struct ConvolutionTester<NSD, cudnn::BackendCUDNN, DataType> {
  ConvolutionTester() {}
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
    Convolution<cudnn::BackendCUDNN, DataType> conv(
        be, 2 + NSD, cfg.halo_exchange_method, cfg.chanfilt_algo);
    conv.setup(d.input, d.filter, d.output,
               d.d_input, d.d_filter, d.d_output,
               cfg.pads,
               cfg.strides,
               cfg.dilations,
               cfg.num_groups,
               cfg.conv_fwd_algo, cfg.conv_bwd_data_algo,
               cfg.conv_bwd_filter_algo, 0, false,
               cfg.deconv);
    if (cfg.use_bias) {
      conv.setup_bias(d.bias);
      conv.setup_bias_gradient(d.d_bias);
    }
    if (pid == 0) {
      std::cout
          << "Forward algorithm: " <<
          util::CUDNNConvolutionFwdAlgorithms::get_name(
              conv.get_fwd_algo())
          << std::endl << "Backward data algorithm: " <<
          util::CUDNNConvolutionBwdDataAlgorithms::get_name(
              conv.get_bwd_data_algo())
          << std::endl << "Backward filter algorithm: " <<
          util::CUDNNConvolutionBwdFilterAlgorithms::get_name(
              conv.get_bwd_filter_algo())
          << std::endl;
    }
    // AUTOTUNE may modify tensors
    if (cfg.conv_fwd_algo == "AUTOTUNE" ||
        cfg.conv_bwd_data_algo == "AUTOTUNE" ||
        cfg.conv_bwd_filter_algo == "AUTOTUNE") {
      d.initialize();
    }
    start_profiler<cudnn::BackendCUDNN>();
    if (cfg.nvtx_marking) {
      be.enable_nvtx_marking();
    }
    test_convolution_forward<NSD, cudnn::BackendCUDNN, DataType>(
      d, cfg, comm, be, conv, prof);
    // These individual tests are mostly redundant as they are also
    // executed as prt of test_convolution_backward.
#if 0
    test_convolution_backward_filter<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, conv, prof);
    test_convolution_backward_data<NSD, cudnn::BackendCUDNN, DataType>(
        d, cfg, comm, be, conv, prof);
    if (cfg.use_bias) {
      test_convolution_backward_bias<NSD, cudnn::BackendCUDNN, DataType>(
          d, cfg, comm, be, conv, prof);
    }
#endif
    test_convolution_backward<NSD, cudnn::BackendCUDNN, DataType>(
      d, cfg, comm, be, conv, prof);
    // This seems necessary to avoid hang using NVSHMEM v0.3.3
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
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

  run_test<NSD, Data, Profile, ConvolutionTester>(cfg, MPI_COMM_WORLD);

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
    util::MPIRootPrintStreamError() << "Invalid --num-dims: " << nsd;
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  Al::Finalize();
  return 0;
}
