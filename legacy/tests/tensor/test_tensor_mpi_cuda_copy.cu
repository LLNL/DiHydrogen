#include "distconv/distconv.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"
#include "test_tensor.hpp"
#include "test_tensor_mpi_cuda_common.hpp"

#include <cmath>
#include <iostream>

using namespace distconv;
using namespace distconv::tensor;

template <>
inline LocaleMPI get_locale<LocaleMPI>() {
  LocaleMPI loc(MPI_COMM_WORLD);
  return loc;
}

template <int ND, typename TensorSrc,
          typename TensorDest>
int test_copy_shuffle(const Array<ND> &shape,
                      const Distribution &dist_src,
                      const Distribution &dist_dest) {
  util::MPIRootPrintStreamInfo() << "test_copy_shuffle";
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert_always(t_src.allocate() == 0);

  int *buf = t_src.get_buffer();
  assert_always(buf || t_src.get_local_shape().is_empty());

  init_tensor<ND><<<4, 4>>>(buf,
                            t_src.get_local_shape(),
                            dist_src.get_overlap(),
                            t_src.get_pitch(),
                            t_src.get_shape(),
                            t_src.get_global_index());

  h2::gpu::sync();

  assert_always(t_dest.allocate() == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "t_dest allocated";

  assert_always(Copy(t_dest, t_src) == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Copy done";

  int error_counter = 0;
  int *error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  h2::gpu::mem_copy(error_counter_d, &error_counter);

  check_tensor<ND><<<1, 1>>>(t_dest.get_const_buffer(),
                             t_dest.get_local_shape(),
                             dist_dest.get_overlap(),
                             t_dest.get_pitch(),
                             t_dest.get_shape(),
                             t_dest.get_global_index(),
                             error_counter_d);
  h2::gpu::mem_copy(&error_counter, error_counter_d);
  assert_always(error_counter == 0);
  return 0;
}

template <int ND, typename TensorSrc,
          typename TensorDest>
int test_copy_shuffle_from_host_to_device(
    const Array<ND> &shape,
    const Distribution &dist_src,
    const Distribution &dist_dest) {
  util::MPIRootPrintStreamInfo() << "test_copy_shuffle";

  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert_always(t_src.allocate() == 0);

  const auto local_shape = t_src.get_local_shape();
  int *buf = t_src.get_buffer();
  assert_always(buf);
  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    index_t x = get_linearlized_offset(t_src.get_global_index(*it),
                                       t_src.get_shape());
    buf[t_src.get_local_offset(*it)] = x;
  }

  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);
  assert_always(t_dest.allocate() == 0);
  assert_always(Copy(t_dest, t_src) == 0);

  int error_counter = 0;
  int *error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  h2::gpu::mem_copy(error_counter_d, &error_counter);

  check_tensor<ND><<<1, 1>>>(t_dest.get_const_buffer(),
                             t_dest.get_local_shape(),
                             dist_dest.get_overlap(),
                             t_dest.get_pitch(),
                             t_dest.get_shape(),
                             t_dest.get_global_index(),
                             error_counter_d);
  h2::gpu::mem_copy(&error_counter, error_counter_d);
  assert_always(error_counter == 0);

  return 0;
}

template <int ND, typename TensorSrc,
          typename TensorDest>
int test_copy_shuffle_from_device_to_host(const Array<ND> &shape,
                                          const Distribution &dist_src,
                                          const Distribution &dist_dest) {
  util::MPIRootPrintStreamInfo() << "test_copy_shuffle";
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert_always(t_src.allocate() == 0);

  int *buf = t_src.get_buffer();
  assert_always(buf);

  init_tensor<ND><<<4, 4>>>(buf,
                            t_src.get_local_shape(),
                            dist_src.get_overlap(),
                            t_src.get_pitch(),
                            t_src.get_shape(),
                            t_src.get_global_index());

  h2::gpu::sync();

  assert_always(t_dest.allocate() == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "t_dest allocated";

  assert_always(Copy(t_dest, t_src) == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Copy done";

  auto local_shape = t_dest.get_local_shape();
  buf = t_dest.get_buffer();
  for (auto it = local_shape.index_begin();
       it != local_shape.index_end(); ++it) {
    int ref = get_linearlized_offset(t_dest.get_global_index(*it),
                                     t_dest.get_shape());
    int stored = buf[t_dest.get_local_offset(*it)];
    if (ref != stored) {
      util::MPIPrintStreamError()
          << "Mismatch at: " << *it
          << ", ref: " << ref << ", stored: " << stored;
      return -1;
    }
  }
  return 0;
}

template <int ND>
void test(const Shape &proc_dim, const Shape &shape) {
  using DataType = int;
  using TensorMPI = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  using TensorHost = Tensor<DataType, LocaleMPI, BaseAllocator>;
  using TensorHostPinned = Tensor<DataType, LocaleMPI,
                                  CUDAHostPooledAllocator>;

  const auto sample_dist = make_sample_distribution(ND, proc_dim.size());
  IntVector overlap(ND, 1);
  overlap[get_sample_dim()] = 0;
  overlap[get_channel_dim()] = 0;

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: local copy from host to device.";
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = Distribution::make_distribution(proc_dim);
    assert0(test_copy_shuffle_from_host_to_device<ND, TensorHost, TensorMPI>(
        shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle_from_device_to_host<ND, TensorMPI, TensorHost>(
        shape, dist1, dist2));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: local copy from device to pinned host.";
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = Distribution::make_distribution(proc_dim);
    assert0(test_copy_shuffle_from_host_to_device<ND, TensorHostPinned, TensorMPI>(
        shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle_from_device_to_host<ND, TensorMPI, TensorHostPinned>(
        shape, dist1, dist2));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy between same shape and distribution. no mpi involved.";
    auto dist1 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from tensor with overlap to non-overlap tensor.";
    auto dist1 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    auto dist2 = Distribution::make_distribution(proc_dim);
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2));
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from sample-distributed tensor to spatially-distributed tensor.";
    auto dist1 = sample_dist;
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2));
    util::MPIRootPrintStreamInfo() << "Testing reverse copy";
    // reverse
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif

#if 1
  {
    //
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from sample-distributed tensor to spatially-distributed tensor. non-divisible tensor sizes.";
    auto dist1 = sample_dist;
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    Array<ND> shape;
    if (ND == 3) {
      shape = {7, 9, 4};
    } else if (ND == 4) {
      shape = {7, 9, 5, 3};
    }
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif


  // cudaMalloc2D does not work with Spectrum MPI. Fails when
  // deallocating with cudaFree.
#if 0
  {
    // From sample to pitched spatial distribution
    auto dist1 = sample_dist;
    auto dist2 = Distribution::make_distribution(proc_dim);
    Array<ND> shape = {proc_x * 2, proc_y * 2, np};
    using TensorMPIPitch = Tensor<ND, DataType, LocaleMPI, CUDAPitchedAllocator>;
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPIPitch>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<ND, TensorMPIPitch, TensorMPI>(shape, dist2, dist1));
  }
#endif
  // cudaMalloc2D does not work with Spectrum MPI. Fails when
  // deallocating with cudaFree.
#if 0
  {
    // local copy with different overlap and pitch
    auto dist1 = Distribution::make_distribution(proc_dim);
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, overlap);
    using TensorMPIPitch = Tensor<ND, DataType, LocaleMPI, CUDAPitchedAllocator>;
    assert0(test_copy_shuffle<ND, TensorMPI, TensorMPIPitch>(shape, dist1, dist2));;
    // reverse
    assert0(test_copy_shuffle<ND, TensorMPIPitch, TensorMPI>(shape, dist2, dist1));
  }
#endif
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_cuda_copy px py [pz], where px
  * py * pz == N
 */
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (argc != 3 && argc != 4) {
    if (pid == 0) {
      std::cerr << "Error! Usage: " << argv[0] << " proc_x proc_y [proc_z]\n";
    }
    MPI_Finalize();
    exit(1);
  }

  int proc_x = atoi(argv[1]);
  int proc_y = atoi(argv[2]);

  if (argc == 3) {
    assert_always(proc_x * proc_y == np);
    test<3>(Shape({proc_x, proc_y, 1}), Shape({8, 8, np}));
  } else if (argc == 4) {
    int proc_z = atoi(argv[3]);
    assert_always(proc_x * proc_y * proc_z == np);
    test<4>(Shape({proc_x, proc_y, proc_z, 1}), Shape({8, 8, 8, np}));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  return 0;
}
