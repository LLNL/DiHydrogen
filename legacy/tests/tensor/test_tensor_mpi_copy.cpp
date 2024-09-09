#include "distconv/distconv.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_mpi.hpp"

#include <cmath>
#include <iostream>

#include "test_tensor.hpp"

using namespace distconv;
using namespace distconv::tensor;

template <>
inline LocaleMPI get_locale<LocaleMPI>()
{
  LocaleMPI loc(MPI_COMM_WORLD);
  return loc;
}

template <typename TensorSrc, typename TensorDest>
int test_copy_shuffle(const Shape& shape,
                      const Distribution& dist_src,
                      const Distribution& dist_dest)
{
  util::MPIRootPrintStreamInfo() << "test_copy_shuffle\n";
  assert_eq(shape.num_dims(), 3);
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert_always(t_src.allocate() == 0);

  auto local_shape = t_src.get_local_shape();
  int* buf = t_src.get_buffer();
  assert_always(buf);
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        index_t x = get_linearlized_offset(t_src.get_global_index(idx),
                                           t_src.get_shape());
        buf[t_src.get_local_offset(idx)] = x;
      }
    }
  }

  assert0(t_dest.allocate());
  assert0(Copy(t_dest, t_src));

  util::MPIPrintStreamDebug()
    << "src tensor: " << t_src << ", dest tensor: " << t_dest;

  local_shape = t_dest.get_local_shape();
  buf = t_dest.get_buffer();
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        int ref = get_linearlized_offset(t_dest.get_global_index(idx),
                                         t_dest.get_shape());
        int stored = buf[t_dest.get_local_offset(idx)];
        // fprintf(stderr, "stored: %d\n", stored);
        // util::MPIPrintStreamDebug() << stored  << "@" << idx <<
        //"\n";
#if 0
        util::MPIPrintStreamDebug() << "stored: " << stored;
#endif
        if (ref != stored)
        {
          util::MPIPrintStreamDebug()
            << "Mismatch at: " << idx << ", ref: " << ref
            << ", stored: " << stored;
          return -1;
        }
      }
    }
  }

  return 0;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_copy px py, where px * py == N
 */
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (argc != 3)
  {
    if (pid == 0)
    {
      std::cerr << "Error! Usage: " << argv[0] << " proc_x proc_y\n";
    }
    MPI_Finalize();
    exit(1);
  }

  int proc_x = atoi(argv[1]);
  int proc_y = atoi(argv[2]);
  assert_always(proc_x * proc_y == np);

  constexpr int ND = 3;
  using DataType = int;
  using TensorMPI = Tensor<DataType, LocaleMPI, BaseAllocator>;

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: copy between same shape and distribution. no mpi involved.";
    auto dist1 = Distribution::make_overlapped_distribution(
      Shape({proc_x, proc_y, 1}), IntVector({1, 1, 0}));
    auto dist2 = Distribution::make_overlapped_distribution(
      Shape({proc_x, proc_y, 1}), IntVector({1, 1, 0}));
    Shape shape({8, 8, np});
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: local copy with different overlap.";
    // copy between same shape but with and without overlap
    auto dist1 = Distribution::make_overlapped_distribution(
      Shape({proc_x, proc_y, 1}), IntVector({1, 1, 0}));
    auto dist2 = Distribution::make_distribution({proc_x, proc_y, 1});
    Shape shape({8, 8, np});
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif

#if 1
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
      << "Test: copy from spatial to sample decomposition";
    auto dist1 =
      Distribution::make_overlapped_distribution({1, 2, np / 2}, {0, 1, 0});
    auto dist2 = make_sample_distribution(ND, np);
    assert0((test_copy_shuffle<TensorMPI, TensorMPI>(
      Shape({2, 2, np}), dist1, dist2)));
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    // copy from sample-distributed tensor to spatially-distributed
    // tensor
    auto dist1 = make_sample_distribution(ND, np);
    auto dist2 = Distribution::make_overlapped_distribution({proc_x, proc_y, 1},
                                                            {1, 1, 0});
    Shape shape({8, 8, np});
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<TensorMPI, TensorMPI>(shape, dist2, dist1));
  }
#endif
  // Copy of Memory with BasePitchedAllocator not supported
#if 0
  {
    MPI_Barrier(MPI_COMM_WORLD);
    // Using pitched memory
    auto dist1 = make_sample_distribution(ND, np);
    auto dist2 = Distribution::make_overlapped_distribution({proc_x, proc_y, 1}, {1, 1, 0});
    Shape shape({proc_x * 2, proc_y * 2, np});
    using TensorMPIPitch = Tensor<DataType, LocaleMPI, BasePitchedAllocator<3>>;
    assert0(test_copy_shuffle<TensorMPI, TensorMPIPitch>(shape, dist1, dist2));
    // reverse
    assert0(test_copy_shuffle<TensorMPIPitch, TensorMPI>(shape, dist2, dist1));
  }
#endif
#if 0
  // Copy of Memory with BasePitchedAllocator not supported
  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: local copy with different overlap and pitch.";
    // local copy with different overlap
    auto dist1 = Distribution::make_overlapped_distribution(
        {proc_x, proc_y, 1}, {0, 0, 0});
    auto dist2 = Distribution::make_overlapped_distribution(
        {proc_x, proc_y, 1}, {1, 1, 0});
    Shape shape({8, 8, np});
    using TensorMPIPitch = Tensor<DataType, LocaleMPI, BasePitchedAllocator<8>>;
    assert0(test_copy_shuffle<TensorMPI, TensorMPIPitch>(shape, dist1, dist2));
    // reverse
    assert0((test_copy_shuffle<TensorMPIPitch, TensorMPI>(shape, dist2, dist1));
  }
#endif

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  return 0;
}
