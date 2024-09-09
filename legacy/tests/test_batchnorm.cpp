#include "distconv/distconv.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_mpi.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>
#ifdef DISTCONV_HAS_CUDA
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/util/util_cuda.hpp"
#endif
#ifdef DISTCONV_HAS_CUDNN
#include "distconv/util/util_gpu_dnn.hpp"
#endif
#include "test_common.hpp"

using namespace distconv;

using distconv::tensor::Shape;

using DataType = float;
constexpr int ND = 4;

template <typename Backend>
struct TensorType;

#ifdef DISTCONV_HAS_CUDNN
template <>
struct TensorType<cudnn::BackendCUDNN>
{
  using type =
    tensor::Tensor<DataType, tensor::LocaleMPI, tensor::CUDAAllocator>;
};
#endif

template <>
struct TensorType<ref::Backend>
{
  using type =
    tensor::Tensor<DataType, tensor::LocaleMPI, tensor::BaseAllocator>;
};

template <typename Backend>
class Data
{
public:
  typename TensorType<Backend>::type input;
  typename TensorType<Backend>::type d_input;
  typename TensorType<Backend>::type output;
  typename TensorType<Backend>::type d_output;
  typename TensorType<Backend>::type mean;
  typename TensorType<Backend>::type var;
  ;
  typename TensorType<Backend>::type running_mean;
  typename TensorType<Backend>::type running_var;
  ;
  typename TensorType<Backend>::type scale;
  typename TensorType<Backend>::type bias;
  typename TensorType<Backend>::type d_scale;
  typename TensorType<Backend>::type d_bias;
  typename TensorType<Backend>::type d_mean;
  typename TensorType<Backend>::type d_var;
};

template <typename Backend>
int setup(test::Config const& cfg, MPI_Comm comm, Data<Backend>& d)
{
  using Tensor = typename TensorType<Backend>::type;

  int pid;
  int np;
  MPI_Comm_rank(comm, &pid);
  MPI_Comm_size(comm, &np);

  Shape input_shape({cfg.i_w, cfg.i_h, cfg.i_c, cfg.i_n});
  // Overlap is not necessary; just for testing
  IntVector overlap({1, 1, 0, 0});

  auto dist = tensor::Distribution::make_overlapped_distribution(
    tensor::Shape({cfg.p_w, cfg.p_h, cfg.p_c, cfg.p_n}), overlap);
  auto shared_dist = tensor::Distribution::make_shared_distribution(
    {cfg.p_w, cfg.p_h, cfg.p_c, cfg.p_n}, {1, 1, cfg.p_c, 1});

  tensor::LocaleMPI loc(comm);
  d.input = Tensor(input_shape, loc, dist);
  d.d_input = Tensor(input_shape, loc, dist);
  d.output = Tensor(input_shape, loc, dist);
  d.d_output = Tensor(input_shape, loc, dist);
  d.mean = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.var = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.running_mean = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.running_var = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.scale = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.bias = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.d_scale = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.d_bias = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.d_mean = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);
  d.d_var = Tensor(Shape({1, 1, cfg.i_c, 1}), loc, shared_dist);

  // Allocate
  assert0(d.input.allocate());
  d.input.zero();
  assert0(d.output.allocate());
  d.output.zero();
  assert0(d.d_input.allocate());
  d.d_input.zero();
  assert0(d.d_output.allocate());
  d.d_output.zero();
  assert0(d.mean.allocate());
  d.mean.zero();
  assert0(d.var.allocate());
  d.var.zero();
  assert0(d.running_mean.allocate());
  d.running_mean.zero();
  assert0(d.running_var.allocate());
  d.running_var.zero();
  assert0(d.scale.allocate());
  d.scale.zero();
  assert0(d.bias.allocate());
  d.bias.zero();
  assert0(d.d_scale.allocate());
  d.d_scale.zero();
  assert0(d.d_bias.allocate());
  d.d_bias.zero();
  assert0(d.d_mean.allocate());
  d.d_mean.zero();
  assert0(d.d_var.allocate());
  d.d_var.zero();

  // Initialization
  test::init_tensor_random(d.input, 0);
  test::init_tensor_random(d.d_output, 1);
  test::init_tensor_constant(d.running_var, 1);
  test::init_tensor_constant(d.scale, 1);

  return 0;
}

template <typename Backend>
int test_forward(Data<Backend>& d,
                 test::Config const& cfg,
                 MPI_Comm comm,
                 Backend& be)
{
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
    << "Executing test_forward with " << be.get_name();

  // Assumes 4-dimensional tensor
  BatchNormalization<Backend, 4, DataType> bn(
    be, 0.9, 1e-5, std::vector<bool>(4, cfg.use_global_stat));
  bn.set_num_samples(d.input.get_shape()[-1]);
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  bn.forward(d.input,
             d.mean,
             d.var,
             d.running_mean,
             d.running_var,
             d.scale,
             d.bias,
             d.output,
             cfg.use_global_stat);
  be.wait();
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Test done";

  return 0;
}

template <typename Backend>
int test_backward(Data<Backend>& d,
                  test::Config const& cfg,
                  MPI_Comm comm,
                  Backend& be)
{
  int pid;
  MPI_Comm_rank(comm, &pid);

  util::MPIRootPrintStreamInfo()
    << "Executing test_backward with " << be.get_name();

  BatchNormalization<Backend, 4, DataType> bn(
    be, 0.9, 1e-5, std::vector<bool>(4, cfg.use_global_stat));

  bn.set_num_samples(d.input.get_shape()[-1]);
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  bn.backward_stage1(d.input,
                     d.d_output,
                     d.mean,
                     d.var,
                     d.scale,
                     d.d_scale,
                     d.d_bias,
                     d.d_mean,
                     d.d_var,
                     cfg.use_global_stat);
  bn.backward_stage2(
    d.input, d.d_output, d.mean, d.var, d.scale, d.d_mean, d.d_var, d.d_input);
  be.wait();
  DISTCONV_CHECK_MPI(MPI_Barrier(comm));
  util::MPIRootPrintStreamInfo() << "Test done";

  return 0;
}

template <typename Backend>
int test_all(Data<Backend>& d, test::Config const& cfg, MPI_Comm comm);

#ifdef DISTCONV_HAS_CUDNN
template <>
int test_all<cudnn::BackendCUDNN>(Data<cudnn::BackendCUDNN>& d,
                                  const test::Config& cfg,
                                  MPI_Comm comm)
{
  int pid;
  DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  cudnnHandle_t cudnn_h;
  DISTCONV_CHECK_CUDNN(cudnnCreate(&cudnn_h));
  cudnn::BackendCUDNN be(comm, cudnn_h);
  test_forward<cudnn::BackendCUDNN>(d, cfg, comm, be);
  test_backward<cudnn::BackendCUDNN>(d, cfg, comm, be);
  be.wait();
  return 0;
}
#endif

#if 0
template <>
int test_all<ref::Backend>(Data<ref::Backend> &d,
                           const test::Config &cfg,
                           MPI_Comm comm) {
  ref::Backend be;
  test_forward<ref::Backend>(d, cfg, comm, be);
  be.wait();
  return 0;
}
#endif

template <typename Backend>
int run(const test::Config& cfg, MPI_Comm comm)
{
  int pid;
  int np;
  DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  DISTCONV_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &np));

  Data<Backend> d;
  setup<Backend>(cfg, MPI_COMM_WORLD, d);

  // Dump input and filter tensors
  if (cfg.dump_input)
  {
    util::MPIRootPrintStreamDebug() << "Dumping input tensors";
    dump_tensor(d.input, "input_tensor");
    dump_tensor(d.d_output, "d_output_tensor");
  }

  util::MPIRootPrintStreamDebug() << "Start testing";
  test_all<Backend>(d, cfg, MPI_COMM_WORLD);

  util::MPIRootPrintStreamDebug() << "Testing done";

  // Dump result
  if (cfg.dump_output)
  {
    dump_tensor(d.output, "output_tensor");
    dump_tensor(d.d_input, "d_input_tensor");
    test::dump_shared_tensor(d.mean, "mean_tensor");
    test::dump_shared_tensor(d.var, "var_tensor");
    test::dump_shared_tensor(d.running_mean, "running_mean_tensor");
    test::dump_shared_tensor(d.running_var, "running_var_tensor");
    test::dump_shared_tensor(d.d_scale, "d_scale_tensor");
    test::dump_shared_tensor(d.d_bias, "d_bias_tensor");
    test::dump_shared_tensor(d.d_mean, "d_mean_tensor");
    test::dump_shared_tensor(d.d_var, "d_var_tensor");
  }

  return 0;
}

int main(int argc, char* argv[])
{
  int pid;
  int np;
  DISTCONV_CHECK_MPI(MPI_Init(&argc, &argv));
  DISTCONV_CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &pid));
  DISTCONV_CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &np));

  test::Config cfg = test::process_opt(argc, argv, pid);
  if (pid == 0)
  {
    std::cout << cfg << std::endl;
  }

  if (cfg.p_n * cfg.p_c * cfg.p_h * cfg.p_w != np)
  {
    util::MPIRootPrintStreamError()
      << "Number of ranks does not match with the number of tensor partitions";
    DISTCONV_CHECK_MPI(MPI_Finalize());
    std::exit(1);
  }

  // Need reduction to validate the results.
  cfg.use_global_stat = true;

  if (cfg.backend == "Ref")
  {
    // run<ref::Backend>(cfg, MPI_COMM_WORLD);
    util::MPIRootPrintStreamError() << "Ref backend not implemented";
#ifdef DISTCONV_HAS_CUDNN
  }
  else if (cfg.backend == "CUDNN")
  {
    int dev = util::choose_gpu();
    util::MPIPrintStreamDebug() << "Using GPU " << dev;
    DISTCONV_CHECK_CUDA(cudaSetDevice(dev));
    run<cudnn::BackendCUDNN>(cfg, MPI_COMM_WORLD);
#endif
  }
  else
  {
    util::MPIRootPrintStreamError() << "Unknown backend name";
    abort();
  }

  util::MPIRootPrintStreamInfo() << "Finishing";

  DISTCONV_CHECK_MPI(MPI_Finalize());
  return 0;
}
