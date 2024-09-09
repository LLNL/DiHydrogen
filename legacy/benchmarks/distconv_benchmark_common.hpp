#pragma once

#include "distconv_config.hpp"
#ifdef DISTCONV_HAS_CUDA
#include "benchmark_common_cuda.hpp"
#endif
#include "distconv/distconv.hpp"
#include "distconv/dnn_backend/batchnorm.hpp"
#include "distconv/dnn_backend/convolution.hpp"
#include "distconv/dnn_backend/cross_entropy.hpp"
#include "distconv/dnn_backend/leaky_relu.hpp"
#include "distconv/dnn_backend/mean_squared_error.hpp"
#include "distconv/dnn_backend/pooling.hpp"
#include "distconv/dnn_backend/relu.hpp"
#include "distconv/dnn_backend/softmax.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/cxxopts.hpp"
#include "distconv/util/stopwatch.h"
#include "distconv/util/util.hpp"

#include "benchmark_common.hpp"

/*
  Miscellaneous structures and functions that should be only used for
  benchmarks using Distconv. cudnn_benchmark, e.g., should not used
  this. Common definitions for all benchmarks go into
  benchmark_common.hpp.
 */

namespace distconv_benchmark
{

using namespace distconv;

// using DataType = float;

template <typename Backend, typename DataType>
struct TensorType;

#ifdef DISTCONV_HAS_CUDNN
template <typename DataType>
struct TensorType<distconv::cudnn::BackendCUDNN, DataType>
{
  using type = distconv::tensor::Tensor<DataType,
                                        distconv::tensor::LocaleMPI,
                                        distconv::tensor::CUDAAllocator>;
};
#endif

template <typename DataType>
struct TensorType<distconv::ref::Backend, DataType>
{
  using type = distconv::tensor::Tensor<DataType,
                                        distconv::tensor::LocaleMPI,
                                        distconv::tensor::BaseAllocator>;
};

inline void set_device()
{
#ifdef DISTCONV_HAS_CUDA
  int dev = distconv::util::choose_gpu();
  DISTCONV_CHECK_CUDA(cudaSetDevice(dev));
#endif
}

template <typename Tensor>
inline int init_input_tensor(Tensor& t, unsigned seed)
{
  using data_type = typename Tensor::data_type;

  // Halo region must be set to zero
  t.zero();
  auto local_shape = t.get_local_shape();
  auto* buf = t.get_buffer();
  assert_always(buf || t.get_local_size() == 0);

  if (buf)
  {
    size_t buf_size = t.get_local_real_shape().get_size() * sizeof(data_type);
    auto* host = (data_type*) calloc(buf_size, 1);

    Initializer<data_type> init(seed);
    auto global_shape = t.get_shape();
    index_t num_local_elms = local_shape.get_size();
#pragma omp parallel for
    for (index_t i = 0; i < num_local_elms; ++i)
    {
      auto local_idx = local_shape.get_index(i);
      auto global_index = t.get_global_index(local_idx);
      host[t.get_local_offset(local_idx)] =
        init.get_initial_value(global_index, global_shape);
    }
    t.copyin(host);
  }
  return 0;
}

template <typename Tensor>
inline int init_tensor_random(Tensor& t)
{
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto* buf = t.get_buffer();
  assert_always(buf || t.get_local_size() == 0);

  if (buf)
  {
    size_t buf_size = t.get_local_real_shape().get_size() * sizeof(data_type);
    auto* host = (data_type*) malloc(buf_size);
    t.copyout(host);

    for (auto it = local_shape.index_begin(); it != local_shape.index_end();
         ++it)
    {
      host[t.get_local_offset(*it)] = (float) (rand()) / RAND_MAX;
    }

    t.copyin(host);
  }
  return 0;
}

template <typename Tensor>
inline int init_tensor_offset(Tensor& t, int start = 0)
{
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto* buf = t.get_buffer();
  assert_always(buf || t.get_local_size() == 0);

  if (buf)
  {
    size_t buf_size = t.get_local_real_shape().get_size() * sizeof(data_type);
    auto* host = (data_type*) malloc(buf_size);
    t.copyout(host);

    for (auto it = local_shape.index_begin(); it != local_shape.index_end();
         ++it)
    {
      // CUDA type half does not accept size_t for casting
      double v = t.get_global_offset(*it) + start;
      host[t.get_local_offset(*it)] = (data_type) (v);
    }

    t.copyin(host);
  }
  return 0;
}

template <typename Tensor>
inline int init_tensor_constant(Tensor& t, typename Tensor::data_type x)
{
  using data_type = typename Tensor::data_type;

  t.zero();
  auto local_shape = t.get_local_shape();
  auto* buf = t.get_buffer();
  assert_always(buf || t.get_local_size() == 0);

  if (buf)
  {
    size_t buf_size = t.get_local_real_shape().get_size() * sizeof(data_type);
    auto* host = (data_type*) malloc(buf_size);
    t.copyout(host);

    for (auto it = local_shape.index_begin(); it != local_shape.index_end();
         ++it)
    {
      host[t.get_local_offset(*it)] = x;
    }

    t.copyin(host);
  }
  return 0;
}

template <typename DataType, typename Alloccator>
inline int dump_shared_tensor(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Alloccator>& t,
  const std::string& file_path,
  bool binary)
{
  if (t.get_locale().get_rank() == 0)
  {
    DataType* buf = (DataType*) malloc(t.get_size() * sizeof(DataType));
    t.get_data().copyout(buf);
    std::ofstream out;
    if (binary)
    {
      out.open(file_path + ".out",
               std::ios::out | std::ios::trunc | std::ios::binary);
      out.write((char*) buf, t.get_size() * sizeof(DataType));
    }
    else
    {
      out.open(file_path + ".txt", std::ios::out | std::ios::trunc);
      for (size_t i = 0; i < t.get_size(); ++i)
      {
        auto x = buf[i];
        out << x << std::endl;
      }
    }
    out.close();
  }
  return 0;
}

template <typename Backend>
struct Clock;

template <>
struct Clock<ref::Backend>
{
  ref::Backend m_be;
  util::stopwatch_t m_st;
  float m_elapsed;
  Clock(ref::Backend& be) : m_be(be), m_elapsed(0) {}
  void start() { util::stopwatch_start(&m_st); }
  void stop() { m_elapsed = util::stopwatch_stop(&m_st); }
  float get_time() { return m_elapsed; }
};

#ifdef DISTCONV_HAS_CUDNN
template <>
struct Clock<cudnn::BackendCUDNN>
{
  cudnn::BackendCUDNN& m_be;
  cudaEvent_t m_ev1;
  cudaEvent_t m_ev2;
  Clock(cudnn::BackendCUDNN& be) : m_be(be)
  {
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_ev1));
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_ev2));
  }
  void start()
  {
    DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev1, m_be.get_stream()));
  }
  void stop()
  {
    DISTCONV_CHECK_CUDA(cudaEventRecord(m_ev2, m_be.get_stream()));
  }
  float get_time()
  {
    DISTCONV_CHECK_CUDA(cudaEventSynchronize(m_ev2));
    float elapsed = 0;
    DISTCONV_CHECK_CUDA(cudaEventElapsedTime(&elapsed, m_ev1, m_ev2));
    return elapsed;
  }
};
#endif

template <typename Backend>
inline void complete_async()
{}

#ifdef DISTCONV_HAS_CUDNN
template <>
inline void complete_async<cudnn::BackendCUDNN>()
{
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
}
#endif

template <typename Backend>
inline void spin_async_device(int ms, Backend& be)
{}

#ifdef DISTCONV_HAS_CUDNN
template <>
inline void spin_async_device<cudnn::BackendCUDNN>(int ms,
                                                   cudnn::BackendCUDNN& be)
{
  spin_gpu(ms, 0);
}
#endif

template <typename Backend>
inline void start_profiler()
{}
template <typename Backend>
inline void stop_profiler()
{}

#ifdef DISTCONV_HAS_CUDNN
template <>
inline void start_profiler<cudnn::BackendCUDNN>()
{
  DISTCONV_CHECK_CUDA(cudaProfilerStart());
}
template <>
inline void stop_profiler<cudnn::BackendCUDNN>()
{
  DISTCONV_CHECK_CUDA(cudaProfilerStop());
}
#endif

template <int NSD,
          typename Backend,
          typename DataType,
          typename Data,
          typename Profile,
          typename Tester>
inline int run_test_with_type(const BenchmarkConfig<NSD>& cfg, MPI_Comm comm)
{
  int pid;
  int np;
  DISTCONV_CHECK_MPI(MPI_Comm_rank(comm, &pid));
  DISTCONV_CHECK_MPI(MPI_Comm_size(comm, &np));

  Profile prof(cfg);
  Data d(cfg, comm);
  if (d.is_empty())
  {
    util::MPIRootPrintStreamDebug()
      << "No computation is done as input/output tensors are empty.";
    return 0;
  }
  d.initialize();

  // Dump input and filter tensors
  if (cfg.dump_input)
  {
    d.dump_input(cfg.dump_binary);
  }

  util::MPIRootPrintStreamDebug() << "Start benchmarking";
  Tester()(d, cfg, comm, prof);
  util::MPIRootPrintStreamDebug() << "Benchmarking done\n";

  if (pid == 0)
  {
    prof.print_summary(std::cout);
  }

  std::ofstream ofs;
  std::stringstream ss;
  ss << cfg.output_file << "_" << pid << ".txt";
  ofs.open(ss.str(), std::fstream::app);
  prof.print_as_row(ofs);

  // Dump result
  if (cfg.dump_output)
  {
    d.dump_output(cfg.dump_binary);
  }

  return 0;
}

template <int NSD,
          typename Backend,
          template <int, typename, typename> class Data,
          template <int> class Profile,
          template <int, typename, typename> class Tester>
inline int run_test_with_backend(const BenchmarkConfig<NSD>& cfg, MPI_Comm comm)
{
  if (cfg.data_type == BenchmarkDataType::FLOAT)
  {
    return run_test_with_type<NSD,
                              Backend,
                              float,
                              Data<NSD, Backend, float>,
                              Profile<NSD>,
                              Tester<NSD, Backend, float>>(cfg, comm);
  }
  else if (cfg.data_type == BenchmarkDataType::DOUBLE)
  {
    return run_test_with_type<NSD,
                              Backend,
                              double,
                              Data<NSD, Backend, double>,
                              Profile<NSD>,
                              Tester<NSD, Backend, double>>(cfg, comm);
#ifdef DISTCONV_ENABLE_FP16
  }
  else if (cfg.backend == "CUDNN" && cfg.data_type == BenchmarkDataType::HALF)
  {
    return run_test_with_type<NSD,
                              Backend,
                              half,
                              Data<NSD, Backend, half>,
                              Profile<NSD>,
                              Tester<NSD, Backend, half>>(cfg, comm);
#endif
  }
  else
  {
    util::MPIPrintStreamError() << "Unknown data type name\n";
    abort();
  }
}

template <int NSD,
          template <int, typename, typename> class Data,
          template <int> class Profile,
          template <int, typename, typename> class Tester>
inline int run_test(const BenchmarkConfig<NSD>& cfg, MPI_Comm comm)
{
  if (cfg.backend == "Ref")
  {
    return run_test_with_backend<NSD, ref::Backend, Data, Profile, Tester>(
      cfg, MPI_COMM_WORLD);
#ifdef DISTCONV_HAS_CUDNN
  }
  else if (cfg.backend == "CUDNN")
  {
    util::MPIRootPrintStreamInfo()
      << "Using " << util::get_cudnn_version_number_string();
    return run_test_with_backend<NSD,
                                 cudnn::BackendCUDNN,
                                 Data,
                                 Profile,
                                 Tester>(cfg, MPI_COMM_WORLD);
#endif
  }
  else
  {
    util::MPIRootPrintStreamError() << "Unknown backend name";
    abort();
  }
}

} // namespace distconv_benchmark
