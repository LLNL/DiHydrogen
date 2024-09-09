#include "distconv/distconv.hpp"
#include "distconv/tensor/shuffle_mpi.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_mpi.hpp"

#include <omp.h>

#include <cmath>
#include <iostream>

#include "test_tensor.hpp"

using DataType = float;

using namespace distconv;
using namespace distconv::tensor;

MPI_Comm local_comm;
int local_comm_size;

template <>
inline LocaleMPI get_locale<LocaleMPI>()
{
  return LocaleMPI(MPI_COMM_WORLD);
}

template <typename Tensor>
void init_tensor(Tensor& t)
{
  for (auto it = t.get_local_shape().index_begin();
       it != t.get_local_shape().index_end();
       ++it)
  {
    auto global_offset = t.get_global_offset(*it);
    t.set(*it, global_offset);
  }
}

template <typename Tensor>
int check_tensor(Tensor& t)
{
  int num_errors = 0;
#pragma omp parallel for
  for (index_t i = 0; i < t.get_local_size(); ++i)
  {
    auto local_index = t.get_local_shape().get_index(i);
    auto global_offset = t.get_global_offset(local_index);
    auto stored = t.get(local_index);
    if (stored != global_offset)
    {
#pragma omp critical
      {
        ++num_errors;
        util::MPIPrintStreamError()
          << "Error at " << t.get_global_index(local_index)
          << "; ref: " << global_offset << ", stored: " << stored;
      }
    }
  }
  return num_errors;
}

template <int ND, typename TensorSrc, typename TensorDest>
int test_copy_shuffle(const Shape& shape,
                      const Distribution& dist_src,
                      const Distribution& dist_dest,
                      ShuffleMethod method,
                      bool skip_backward = false)
{
  using Allocator = typename TensorSrc::allocator_type;
  auto loc_src = get_locale<typename TensorSrc::locale_type>();
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);

  util::MPIRootPrintStreamDebug()
    << "Transposing " << t_src << " to " << t_dest;

  assert_always(t_src.allocate() == 0);
  for (auto it = t_src.get_local_shape().index_begin();
       it != t_src.get_local_shape().index_end();
       ++it)
  {
    auto global_offset = t_src.get_global_offset(*it);
    t_src.set(*it, global_offset);
  }

  assert_always(t_dest.allocate() == 0);

  TensorMPIShuffler<DataType, Allocator>* shuffler = nullptr;

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Creating a shuffler";

  switch (method)
  {
  case ShuffleMethod::MPI:
    shuffler = new TensorMPIShuffler<DataType, Allocator>(t_src, t_dest);
    break;
  default:
    util::MPIRootPrintStreamError() << "Unknown shuffle method";
    std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Executing shuffle_forward";
  shuffler->shuffle_forward(t_src.get_base_ptr(), t_dest.get_base_ptr());
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Checking results";

  int error_counter = 0;

  if (t_dest.is_split_root())
  {
    error_counter = check_tensor(t_dest);
    util::MPIPrintStreamDebug() << "#errors: " << error_counter;
  }

  MPI_Allreduce(
    MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (error_counter != 0)
  {
    distconv::dump_tensor(t_dest, "tensor_dest", true);
    distconv::dump_tensor(t_src, "tensor_src", true);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  assert0(error_counter);

  if (skip_backward)
    return 0;

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIPrintStreamDebug() << "Transposing " << t_dest << " to " << t_src;

  util::MPIRootPrintStreamInfo() << "Executing shuffle_backward";
  shuffler->shuffle_backward(t_dest.get_base_ptr(), t_src.get_base_ptr());

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamDebug() << "Checking results";

  if (t_src.is_split_root())
  {
    error_counter = check_tensor(t_src);
    if (error_counter)
    {
      util::MPIPrintStreamError() << "#errors: " << error_counter;
    }
  }
  MPI_Allreduce(
    MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (error_counter != 0)
  {
    distconv::dump_tensor(t_dest, "tensor_dest", true);
    distconv::dump_tensor(t_src, "tensor_src", true);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  assert0(error_counter);

  delete shuffler;

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

Distribution get_sample_dist(const Shape& shape, int np)
{
  int last_dim = shape[get_sample_dim()];
  if (last_dim >= np)
  {
    return make_sample_distribution(shape.num_dims(), np);
  }
  assert0(np % last_dim);
  Shape proc_shape(shape.num_dims(), 1);
  proc_shape[get_sample_dim()] = last_dim;
  proc_shape[0] = np / last_dim;
  auto split_shape = proc_shape;
  split_shape[0] = 1;
  auto d = Distribution::make_shared_distribution(proc_shape, split_shape);
  util::MPIRootPrintStreamInfo() << "Using strided sample distribution: " << d;
  return d;
}

template <int ND, typename Allocator>
int run_tests(const Shape& proc_dim, const Shape& shape, ShuffleMethod method)
{
  constexpr int NSD = ND - 2;
  const auto create_spatial_overlap = []() {
    IntVector v(NSD, 1);
    v.push_back(0);
    v.push_back(0);
    return v;
  };
  using TensorMPI = Tensor<DataType, LocaleMPI, Allocator>;
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  util::MPIRootPrintStreamInfo() << "Run tests with " << method;

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: copy between same shape and distribution with no halo.";
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = Distribution::make_distribution(proc_dim);
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2, method))
      == 0);
    // reverse
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1, method))
      == 0);
  }

  // Host shuffler does not support halo
#if 0
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from tensor with overlap to non-overlap tensor.";
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, create_spatial_overlap());
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist1, dist2, method)) == 0);
    // reverse
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist2, dist1, method)) == 0);
  }
#endif

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo() << "Test: copy from sample-distributed "
                                      "tensor to spatially-distributed tensor.";
    auto dist1 = get_sample_dist(shape, np);
    auto dist2 = Distribution::make_distribution(proc_dim);
    util::MPIRootPrintStreamInfo()
      << "dist1 (" << dist1 << ") to dist2 (" << dist2 << ")";
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2, method))
      == 0);
    MPI_Barrier(MPI_COMM_WORLD);
    // reverse
    util::MPIRootPrintStreamInfo()
      << "dist2 (" << dist2 << ") to dist1 (" << dist1 << ")";
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1, method))
      == 0);
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: copy from sample-distributed tensor to spatially-distributed "
         "tensor with halo.";
    auto dist1 = get_sample_dist(shape, np);
    auto dist2 = Distribution::make_overlapped_distribution(
      proc_dim, create_spatial_overlap());
    util::MPIRootPrintStreamInfo()
      << "dist1 (" << dist1 << ") to dist2 (" << dist2 << ")";
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
                    shape, dist1, dist2, method, true))
                  == 0);
    MPI_Barrier(MPI_COMM_WORLD);
    // Only sample-to-spatial is supported for overlapped tensors
#if 0
    // reverse
    util::MPIRootPrintStreamInfo() << "dist2 (" << dist2
                                   << ") to dist1 (" << dist1 << ")";
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist2, dist1, method)) == 0);
#endif
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: shrink distribution of the spatial dimensions.";
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = dist1;
    auto shrunk_split_shape = dist2.get_split_shape();
    for (int i = 0; i < NSD; i++)
      shrunk_split_shape[i] = 1;
    dist2.set_split_shape(shrunk_split_shape);
    util::MPIRootPrintStreamInfo() << "dist1 to dist2";
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2, method))
      == 0);
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo() << "dist2 to dist1";
    assert_always(
      (test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1, method))
      == 0);
  }

#if 0
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: shrink distribution of the spatial dimensions from sample-parallel tensors.";
    auto split_shape = proc_dim;
    for(int i = 0; i < NSD; i++)
      split_shape[i] = 1;
    Distribution dist1(proc_dim, split_shape, create_spatial_overlap(),
                       Shape(proc_dim.num_dims(), 0));
    auto sample_dist = get_sample_dist(shape, np);
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist1, sample_dist, method)) == 0);
    MPI_Barrier(MPI_COMM_WORLD);
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, sample_dist, dist1, method)) == 0);
  }
#endif
  return 0;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_cuda_shuffle pw ph pc pn [w [h [c
  [n]]]], where pw * ph * pc * pn == N with optional h, w, c, n specifying the
  dimensions of a test tensor.
 */
int main(int argc, char* argv[])
{
  const auto pop_arg = [&argc, &argv] {
    const std::string arg(*argv);
    argv++;
    argc--;
    return arg;
  };

  MPI_Init(&argc, &argv);

  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_Comm_split_type(
    MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  MPI_Comm_size(local_comm, &local_comm_size);

  const std::string bin = pop_arg();
  const auto print_usage_and_exit = [bin](const std::string usage) {
    util::MPIRootPrintStreamError() << "Error! Usage: " << bin << " " << usage;
    MPI_Finalize();
    exit(1);
  };

  // Parse the number of spatial dimensions
  if (argc < 1)
    print_usage_and_exit("ND");
  const int NSD = std::stoi(pop_arg());
  if (!(NSD == 2 || NSD == 3))
  {
    util::MPIRootPrintStreamError()
      << "Invalid number of spatial dimensions: " << NSD;
    MPI_Finalize();
    exit(1);
  }
  const int ND = NSD + 2;

  // Parse the proc shape
  std::vector<std::string> dim_names;
  if (ND == 4)
    dim_names = {"w", "h", "c", "n"};
  else
    dim_names = {"w", "h", "d", "c", "n"};
  std::transform(
    dim_names.begin(),
    dim_names.end(),
    dim_names.begin(),
    [](const std::string name) { return std::string("proc_") + name; });
  if (argc < ND)
    print_usage_and_exit("ND " + util::join_spaced_array(dim_names));
  Shape proc_dim_v;
  for (int i = 0; i < ND; i++)
  {
    proc_dim_v.push_back(std::stoi(pop_arg()));
  }

  // Parse the tensor shape
  Shape tensor_shape_v(ND - 2, 8);
  tensor_shape_v.push_back(2);
  tensor_shape_v.push_back(np);
  for (int i = 0; i < ND; i++)
    if (argc > 0)
    {
      tensor_shape_v[i] = std::stoi(pop_arg());
    }

  // Run tests
  std::vector<ShuffleMethod> methods;
  if (argc > 0)
  {
    std::string method_name = pop_arg();
    if (method_name == "MPI")
    {
      methods.push_back(ShuffleMethod::MPI);
    }
    else
    {
      util::MPIRootPrintStreamError() << "Unknown method name: " << method_name;
      MPI_Finalize();
      exit(1);
    }
  }
  else
  {
    methods = {ShuffleMethod::MPI};
  }

  using Allocator = tensor::BaseAllocator;

  MPI_Barrier(MPI_COMM_WORLD);
  for (const auto method : methods)
  {
    if (ND == 4)
    {
      run_tests<4, Allocator>(proc_dim_v, tensor_shape_v, method);
    }
    else
    {
      run_tests<5, Allocator>(proc_dim_v, tensor_shape_v, method);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  return 0;
}
