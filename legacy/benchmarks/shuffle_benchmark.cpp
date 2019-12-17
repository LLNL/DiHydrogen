#include "distconv_benchmark_common.hpp"
#include "benchmark_common.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/tensor/shuffle_mpi.hpp"
#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_al.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/shuffle_mpi_cuda_p2p.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_hybrid.hpp"
#endif // DISTCONV_HAS_P2P
#include "distconv/distconv.hpp"
#include "distconv/util/cxxopts.hpp"
#include "distconv/util/stopwatch.h"

#include <Al.hpp>

#include <iostream>
#include <numeric>
#include <typeinfo>

#define HOST_SKIP_BACKWARD

using DataType = float;
using namespace distconv;
using distconv::tensor::Shape;
using AlBackend = Al::MPICUDABackend;

namespace distconv_benchmark {

template <int NSD>
class Profile {
 public:
  std::vector<float> fwd_time;
  std::vector<float> bwd_time;
  distconv_benchmark::BenchmarkConfig<NSD> m_cfg;
  Profile(const distconv_benchmark::BenchmarkConfig<NSD> &cfg):
      m_cfg(cfg) {}

  std::ostream &print_as_row(std::ostream &os) const {
    std::stringstream ss;
    ss << m_cfg.shuffle_method << " "
       << util::join_spaced_array(m_cfg.i_s)
       << " " << m_cfg.i_c << " " << m_cfg.i_n;
    for (size_t i = 0; i < fwd_time.size(); ++i) {
      os << ss.str() << " fwd " << fwd_time[i] << std::endl;
    }
    for (size_t i = 0; i < bwd_time.size(); ++i) {
      os << ss.str() << " bwd " << bwd_time[i] << std::endl;
    }
    return os;
  }

  void print_summary(std::ostream &os) const {
    if (fwd_time.size() > 0) {
      std::cout << m_cfg.shuffle_method
                << " fwd mean: " << get_mean(fwd_time)
                << ", median: " << get_median(fwd_time)
                << " (ms)\n";
    }
    if (bwd_time.size() > 0) {
      std::cout << m_cfg.shuffle_method
                << " bwd mean: " << get_mean(bwd_time)
                << ", median: " << get_median(bwd_time)
                << " (ms)\n";
    }
  }
};

template <typename Allocator>
class Data {
  using Tensor = tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>;
 public:
  Tensor sample;
  Tensor spatial;
  Tensor output_sample;
};

template <int NSD, typename Allocator>
int setup(const distconv_benchmark::BenchmarkConfig<NSD> &cfg,
          MPI_Comm comm, Data<Allocator> &d) {
  using Tensor = tensor::Tensor<DataType, tensor::LocaleMPI,
                                Allocator>;
  int pid;
  int np;
  MPI_Comm_rank(comm, &pid);
  MPI_Comm_size(comm, &np);

  util::MPIPrintStreamDebug() << "Creating the input tensor";

  tensor::Shape shape(cfg.i_s);
  shape.push_back(cfg.i_c);
  shape.push_back(cfg.i_n);
  tensor::Shape proc_shape(cfg.p_s);
  proc_shape.push_back(cfg.p_c);
  proc_shape.push_back(cfg.p_n);
  IntVector overlap(shape.num_dims(), 0);
  // The host shuffler does not support overlapped tensors
  if (!cfg.host) {
    // Halo size of 1
    for (int i = 0; i < NSD; ++i) {
      overlap[i] = 1;
    }
  }
  auto spatial_dist = tensor::Distribution::make_overlapped_distribution(
      proc_shape, overlap);
  auto sample_dist = make_strided_sample_distribution(
      shape.num_dims(), shape[-1], np);

  d.spatial = Tensor(shape, tensor::LocaleMPI(comm), spatial_dist);
  d.sample = Tensor(shape, tensor::LocaleMPI(comm), sample_dist);
  d.output_sample = Tensor(shape, tensor::LocaleMPI(comm), sample_dist);

  if (pid == 0) {
    std::cout << "Spatial tensor shape: " << shape
              << ", distribution: " << d.spatial.get_distribution() << "\n";
    std::cout << "Sample tensor shape: " << shape
              << ", distribution: " << d.sample.get_distribution() << "\n";
  }

    // Allocate
  assert0(d.sample.allocate());
  d.sample.zero();
  assert0(d.output_sample.allocate());
  d.output_sample.zero();
  assert0(d.spatial.allocate());
  d.spatial.zero();

  return 0;
}

// unused
#if 0
template <int NSD>
distconv_benchmark::BenchmarkConfig<NSD> process_opt(int argc, char *argv[],
                                                     int pid) {
  cxxopts::Options cmd_opts(argv[0], "Tensor Copy Benchmark");
  cmd_opts.add_options()
      ("host", "Run benchmark on host")
      ("cuda", "Run benchmark on GPU")
      ("r,num-runs", "Number of runs", cxxopts::value<int>())
      ("num-warmup-runs", "Number of warming-up runs", cxxopts::value<int>())
      ("o,output-file", "Save performance profile to file", cxxopts::value<std::string>())
      ("h,image-height", "Image height", cxxopts::value<int>())
      ("w,image-width", "Image width", cxxopts::value<int>())
      ("image-size", "Image size", cxxopts::value<int>())
      ("n,num-images", "Number of images", cxxopts::value<int>())
      ("c,num-channels", "Number of channels", cxxopts::value<int>())
      ("p-w", "Number of process groups in the width dimension", cxxopts::value<int>())
      ("p-h", "Number of process groups in the height dimension", cxxopts::value<int>())
      ("p-c", "Number of process groups in the channel dimension", cxxopts::value<int>())
      ("p-n", "Number of process groups in the sample dimension", cxxopts::value<int>())
      ("enable-rma", "Use the RMA method")
      ("enable-MPI", "Use the MPI method")
      ("enable-p2p", "Use the P2P method")
      ("enable-al", "Use the AL method")
      ("enable-hybrid", "Use the HYBRID method")
      ("i,dump-input", "Dump input tensors")
      ("d,dump-output", "Dump output tensors")
      ("dump", "Dump input and output tensors")
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
#endif

template <int NSD>
int test_shuffler(Data<tensor::BaseAllocator> &d,
                  const distconv_benchmark::BenchmarkConfig<NSD> &cfg,
                  MPI_Comm comm,
                  Profile<NSD> &prof) {
  using Allocator = tensor::BaseAllocator;
  util::MPIRootPrintStreamInfo() << "Executing " << __FUNCTION__
                                 << " with " << cfg.shuffle_method;
  using Shuffler = tensor::TensorMPIShuffler<DataType, Allocator>;
  Shuffler *shfl = nullptr;

  DataType *src_buf = static_cast<DataType*>(
      util::aligned_malloc(Shuffler::get_buf_size(d.sample)));
  DataType *dst_buf = static_cast<DataType*>(
      util::aligned_malloc(Shuffler::get_buf_size(d.spatial)));

  switch (cfg.shuffle_method) {
     case ShuffleMethod::MPI:
       shfl = new Shuffler(d.sample, d.spatial, src_buf, dst_buf);
      break;
    default:
      util::MPIRootPrintStreamError() << "Unknown shuffle method";
      std::abort();
  }
  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    shfl->shuffle_forward(d.sample.get_base_ptr(),
                          d.spatial.get_base_ptr());
#ifndef HOST_SKIP_BACKWARD
    shfl->shuffle_backward(d.spatial.get_base_ptr(),
                           d.output_sample.get_base_ptr());
#endif
  }
  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurements";
  util::MPIRootPrintStreamInfo() << "Measuring shuffle_forward";
  for (int i = 0; i < cfg.run_count; ++i) {
    util::stopwatch_t st;
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    util::stopwatch_start(&st);
    shfl->shuffle_forward(d.sample.get_base_ptr(),
                          d.spatial.get_base_ptr());
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    float elapsed = util::stopwatch_stop(&st);
    prof.fwd_time.push_back(elapsed);
  }
#ifndef HOST_SKIP_BACKWARD
  util::MPIRootPrintStreamInfo() << "Measuring shuffle_backward";
  for (int i = 0; i < cfg.run_count; ++i) {
    util::stopwatch_t st;
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    util::stopwatch_start(&st);
    shfl->shuffle_backward(d.spatial.get_base_ptr(),
                           d.output_sample.get_base_ptr());
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    float elapsed = util::stopwatch_stop(&st);
    prof.bwd_time.push_back(elapsed);
  }
#endif
  delete shfl;
  free(src_buf);
  free(dst_buf);
  util::MPIRootPrintStreamInfo() << "Measurement done";
  return 0;
}

template <int NSD>
int test_shuffler(Data<tensor::CUDAAllocator> &d,
                  const distconv_benchmark::BenchmarkConfig<NSD> &cfg,
                  MPI_Comm comm,
                  Profile<NSD> &prof) {
  //using Allocator = tensor::CUDAAllocator;
  util::MPIRootPrintStreamInfo() << "Executing " << __FUNCTION__
                                 << " with " << cfg.shuffle_method;

#ifdef DISTCONV_HAS_P2P
  p2p::P2P *p2p_h = nullptr;
#endif // DISTCONV_HAS_P2P
  AlBackend::comm_type *al_comm = nullptr;
  tensor::TensorMPICUDAShuffler<DataType> *shfl = nullptr;
  cudaStream_t stream;
  DISTCONV_CHECK_CUDA(cudaStreamCreate(&stream));

  switch (cfg.shuffle_method) {
    case ShuffleMethod::AL:
      al_comm = new AlBackend::comm_type(comm, stream);
      shfl = new tensor::TensorMPICUDAShufflerAL<DataType>(
          d.sample, d.spatial, *al_comm);
      break;
    case ShuffleMethod::MPI:
      shfl = new tensor::TensorMPICUDAShuffler<DataType>(
          d.sample, d.spatial);
      break;
#ifdef DISTCONV_HAS_P2P
    case ShuffleMethod::P2P:
      p2p_h = new p2p::P2P(comm);
      shfl = new tensor::TensorMPICUDAShufflerP2P<DataType>(
          d.sample, d.spatial, *p2p_h);
      break;
    case ShuffleMethod::HYBRID:
      al_comm = new AlBackend::comm_type(comm, stream);
      p2p_h = new p2p::P2P(comm);
      shfl = new tensor::TensorMPICUDAShufflerHybrid<DataType>(
          d.sample, d.spatial, *p2p_h, *al_comm);
      break;
#endif // DISTCONV_HAS_P2P
    default:
      util::MPIRootPrintStreamError() << "Unknown shuffle method";
      std::abort();
  }

  if (cfg.warming_up_count > 0) {
    util::MPIRootPrintStreamInfo() << "Warming up";
  }
  for (int i = 0; i < cfg.warming_up_count; ++i) {
    shfl->shuffle_forward(d.sample.get_base_ptr(),
                          d.spatial.get_base_ptr());
    shfl->shuffle_backward(d.spatial.get_base_ptr(),
                           d.output_sample.get_base_ptr());
  }

  util::MPIRootPrintStreamInfo() << "Starting " << cfg.run_count
                                 << " times of measurements";
  util::MPIRootPrintStreamInfo() << "Measuring shuffle_forward";
  std::vector<util::Clock> clks(cfg.run_count, stream);
  for (int i = 0; i < cfg.run_count; ++i) {
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    clks[i].start();
    shfl->shuffle_forward(d.sample.get_base_ptr(),
                          d.spatial.get_base_ptr(),
                          stream);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    prof.fwd_time.push_back(clks[i].get_time());
  }

  util::MPIRootPrintStreamInfo() << "Measuring shuffle_backward";
  for (int i = 0; i < cfg.run_count; ++i) {
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
    DISTCONV_CHECK_MPI(MPI_Barrier(comm));
    clks[i].start();
    shfl->shuffle_backward(d.spatial.get_base_ptr(),
                           d.output_sample.get_base_ptr(),
                           stream);
    clks[i].stop();
  }
  DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
  for (int i = 0; i < cfg.run_count; ++i) {
    prof.bwd_time.push_back(clks[i].get_time());
  }

  delete shfl;
#ifdef DISTCONV_HAS_P2P
  if (p2p_h) delete p2p_h;
#endif // DISTCONV_HAS_P2P
  if (al_comm) delete al_comm;
  DISTCONV_CHECK_CUDA(cudaStreamDestroy(stream));

  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Measurement done";

  return 0;
}

template <int NSD>
void dump_prof(const Profile<NSD> &prof, int pid,
               const distconv_benchmark::BenchmarkConfig<NSD> &cfg) {
  if (pid == 0) {
    prof.print_as_row(std::cout);
    prof.print_summary(std::cout);
    std::cout.flush();
  }
}

template <int NSD, typename Allocator>
int run_test(const distconv_benchmark::BenchmarkConfig<NSD> &cfg, MPI_Comm comm) {
  int pid;
  int np;
  DISTCONV_CHECK_MPI(MPI_Comm_rank(comm, &pid));
  DISTCONV_CHECK_MPI(MPI_Comm_size(comm, &np));

  Data<Allocator> d;
  setup<NSD, Allocator>(cfg, comm, d);

  if (cfg.dump_input) {
    util::MPIPrintStreamDebug() << "Dumping input tensors";
    dump_tensor(d.sample, "input_sample_tensor", true);
  }

  Profile<NSD> prof(cfg);
  d.spatial.zero();
  d.output_sample.zero();
  test_shuffler<NSD>(d, cfg, comm, prof);
  dump_prof(prof, pid, cfg);

  if (cfg.dump_output) {
    dump_tensor(d.spatial, "output_spatial_tensor", true);
    dump_tensor(d.output_sample, "output_sample_tensor", true);
  }
  return 0;
}

template <int NSD>
void run(int argc, char *argv[], int pid, int np) {
  auto cfg = distconv_benchmark::process_opt<NSD>(argc, argv, pid, true);
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

  if (cfg.p_c != 1) {
    util::MPIRootPrintStreamError()
        << "Partitioning channel dimension not supported";
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  if (cfg.host) {
    // Only MPI is supported for host tensors
    cfg.shuffle_method = ShuffleMethod::MPI;
    run_test<NSD, tensor::BaseAllocator>(cfg, MPI_COMM_WORLD);
  } else {
    run_test<NSD, tensor::CUDAAllocator>(cfg, MPI_COMM_WORLD);
  }

  util::MPIRootPrintStreamInfo() << "Completed";
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
    util::MPIRootPrintStreamError()
        << "Invalid --num-dims: " << nsd;
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  Al::Finalize();
  return 0;
}
