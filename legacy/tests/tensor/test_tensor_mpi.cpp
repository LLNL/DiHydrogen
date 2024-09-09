#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_mpi.hpp"

#include <cassert>
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

template <typename TensorDest, typename TensorSrc>
int test_view(const Shape& shape,
              const Distribution& dist_src,
              const Distribution& dist_dest)
{
  util::MPIPrintStreamInfo() << "test_view";

  assert_eq(shape.num_dims(), 3);

  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert0(t_src.allocate());

  auto local_shape = t_src.get_local_shape();
  auto* buf = t_src.get_buffer();
  assert_always(buf || local_shape.get_size() == 0);
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

  auto t_view = get_tensor<TensorDest>(loc_dest, dist_dest);
  View(t_view, t_src);
  util::MPIRootPrintStreamDebug() << "const view created\n";

  local_shape = t_view.get_local_shape();
  const auto* view_buf = t_view.get_buffer();
  assert_always(view_buf || local_shape.get_size() == 0);
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        int ref = get_linearlized_offset(t_src.get_global_index(idx),
                                         t_src.get_shape());
        int stored = view_buf[t_view.get_local_offset(idx)];
        if (ref != stored)
        {
          std::cerr << "Mismatch at: " << idx << ", ref: " << ref
                    << ", stored: " << stored << "\n";
          return -1;
        }
      }
    }
  }

  return 0;
}

template <typename TensorType>
int test_view_raw_ptr(const Shape& shape, const Distribution& dist)
{
  assert_eq(shape.num_dims(), 3);
  auto loc = get_locale<typename TensorType::locale_type>();
  auto t = get_tensor<TensorType>(shape, loc, dist);
  assert0(t.allocate());

  auto local_shape = t.get_local_shape();
  auto* buf = t.get_buffer();
  assert_always(buf || t.get_local_size() == 0);
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        index_t x =
          get_linearlized_offset(t.get_global_index(idx), t.get_shape());
        buf[t.get_local_offset(idx)] = x;
      }
    }
  }

  using ConstTensorType = Tensor<typename TensorType::data_type,
                                 typename TensorType::locale_type,
                                 typename TensorType::allocator_type>;
  auto const_tensor_view = get_tensor<ConstTensorType>(shape, loc, dist);
  View(const_tensor_view, (const typename ConstTensorType::data_type*) buf);
  assert_always(const_tensor_view.get_const_buffer() == buf);

  const auto* view_buf = const_tensor_view.get_const_buffer();
  assert_always(view_buf || const_tensor_view.get_local_size() == 0);
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        int ref =
          get_linearlized_offset(t.get_global_index(idx), t.get_shape());
        int stored = view_buf[const_tensor_view.get_local_offset(idx)];
        if (ref != stored)
        {
          std::cerr << "Mismatch at: " << idx << ", ref: " << ref
                    << ", stored: " << stored << "\n";
          return -1;
        }
      }
    }
  }

  return 0;
}

template <typename TensorDest, typename TensorSrc>
int test_copy(const Shape& shape,
              const Distribution& dist_src,
              const Distribution& dist_dest,
              int root)
{
  assert_eq(shape.num_dims(), 3);
  util::MPIRootPrintStreamInfo() << "test_copy\n";
  util::MPIRootPrintStreamInfo() << "Distribution dest: " << dist_dest << "\n";

  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_dest = get_tensor<TensorDest>(loc_dest, dist_dest);
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert0(t_src.allocate());

  auto local_shape = t_src.get_local_shape();
  auto* buf = t_src.get_buffer();
  assert_always(buf || t_src.get_local_size() == 0);
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

  assert0(Copy(t_dest, t_src, root));

  if (loc_src.get_rank() == root)
  {
    local_shape = t_dest.get_local_shape();
    buf = t_dest.get_buffer();
    for (index_t i = 0; i < local_shape[2]; ++i)
    {
      for (index_t j = 0; j < local_shape[1]; ++j)
      {
        for (index_t k = 0; k < local_shape[0]; ++k)
        {
          IndexVector idx({k, j, i});
          int ref = get_linearlized_offset(t_src.get_global_index(idx),
                                           t_src.get_shape());
          int stored = buf[t_dest.get_local_offset(idx)];
          util::MPIPrintStreamDebug() << stored << "@" << idx << "\n";
          if (ref != stored)
          {
            std::cerr << "Mismatch at: " << idx << ", ref: " << ref
                      << ", stored: " << stored << "\n";
            return -1;
          }
        }
      }
    }
  }

  return 0;
}

template <typename TensorDest, typename TensorSrc>
int test_copy_redist(const Shape& shape,
                     const Distribution& dist_src,
                     const Distribution& dist_dest)
{
  assert_eq(shape.num_dims(), 3);
  std::cerr << "test_copy_redist\n";
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto loc_src = get_locale<typename TensorSrc::locale_type>();

  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  assert0(t_src.allocate());

  auto local_shape = t_src.get_local_shape();
  int* buf = t_src.get_buffer();
  assert_always(buf || t_src.get_local_size() == 0);
  for (index_t i = 0; i < local_shape[2]; ++i)
  {
    for (index_t j = 0; j < local_shape[1]; ++j)
    {
      for (index_t k = 0; k < local_shape[0]; ++k)
      {
        IndexVector idx({k, j, i});
        index_t x = get_linearlized_offset(t_src.get_global_index({k, j, i}),
                                           t_src.get_shape());
        buf[t_src.get_local_offset(idx)] = x;
      }
    }
  }

  assert0(t_dest.allocate());

  assert0(CopyX(t_dest, t_src));

  local_shape = t_dest.get_local_shape();
  buf = t_dest.get_buffer();
  assert_always(buf || t_dest.get_local_size() == 0);
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
        // util::MPIPrintStreamDebug() << stored  << "@" << idx << "\n";
        if (ref != stored)
        {
          util::MPIPrintStreamDebug()
            << "Mismatch at: " << idx << ", ref: " << ref
            << ", stored: " << stored << "\n";
          return -1;
        }
      }
    }
  }

  return 0;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi, where N must be >= 8 and
  divisible by 8.
 */
int main(int argc, char* argv[])
{
  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  using DataType = int;
  using LocaleMPI = LocaleMPI;
  using LocaleProc = LocaleProcess;
  using TensorMPI = Tensor<DataType, LocaleMPI, BaseAllocator>;
  using TensorProc = Tensor<DataType, LocaleProc, BaseAllocator>;

  assert_always((np % 2) == 0 && np >= 2);

  auto dist =
    Distribution::make_overlapped_distribution({1, 2, np / 2}, {0, 0, 1});

  assert0(test_data_access<TensorMPI>(Shape({2, 2, 2}), dist));
  util::MPIRootPrintStreamInfo() << "test_data_access success";

  auto shared_dist = Distribution::make_localized_distribution(3);
  util::MPIRootPrintStreamDebug() << "shared_dist: " << shared_dist;

  assert0(test_data_access<TensorMPI>(Shape({2, 2, 2}), shared_dist));

  util::MPIRootPrintStreamInfo()
    << "test_data_access with a shared tensor success";

  assert0(
    test_view<TensorProc, TensorMPI>(Shape({2, 2, 2}), dist, shared_dist));
  util::MPIRootPrintStreamInfo() << "test_view success";

  assert0(test_view_raw_ptr<TensorMPI>(Shape({2, 2, 2}), dist));
  util::MPIRootPrintStreamInfo() << "test_view_raw_ptr success";

  assert0(
    test_copy<TensorProc, TensorMPI>(Shape({2, 2, 4}), dist, shared_dist, 0));
  util::MPIRootPrintStreamInfo() << "test_copy success";

  util::MPIRootPrintStreamInfo() << "Testing 4D tensors";
  assert_always((np % 8) == 0 && np >= 8);
  using TensorMPI4 = Tensor<DataType, LocaleMPI, BaseAllocator>;
  auto dist4 = Distribution::make_overlapped_distribution(
    Shape({2, 2, 2, np / 8}), IntVector({1, 1, 0, 0}));

  assert0(test_alloc<TensorMPI4>(Shape({2, 2, 2, 2}), dist4));
  util::MPIRootPrintStreamInfo() << "test_alloc success";

  assert0(test_data_access4<TensorMPI4>(Shape({2, 2, 2, 2}), dist4));
  util::MPIRootPrintStreamInfo() << "test_data_access success";

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  return 0;
}
