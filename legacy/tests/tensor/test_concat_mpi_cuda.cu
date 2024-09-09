#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

#include <iostream>
#include <vector>

#include "test_tensor.hpp"

using namespace distconv;
using namespace distconv::tensor;
using namespace distconv::util;

template <>
inline LocaleMPI get_locale()
{
  LocaleMPI loc(MPI_COMM_WORLD);
  return loc;
}

template <typename TensorType>
inline int test_concat(const Shape& dst_shape,
                       const Distribution& dst_dist,
                       const Shape& src1_shape,
                       const Shape& src2_shape,
                       const Distribution& src_dist)
{
  const int num_dims = dst_shape.num_dims();
  using DataType = typename TensorType::data_type;
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType dst = get_tensor<TensorType>(dst_shape, loc, dst_dist);
  TensorType src1 = get_tensor<TensorType>(src1_shape, loc, src_dist);
  TensorType src2 = get_tensor<TensorType>(src2_shape, loc, src_dist);

  assert0(dst.allocate());
  assert0(src1.allocate());
  assert0(src2.allocate());

  dst.zero();
  src1.zero();
  src2.zero();

  int concat_dim = -1;
  for (int i = 0; i < num_dims; ++i)
  {
    if (dst_shape[i] == src1_shape[i] && dst_shape[i] == src2_shape[i])
    {
      continue;
    }
    else
    {
      assert_always(dst_shape[i] == src1_shape[i] + src2_shape[i]);
      concat_dim = i;
    }
  }
  assert_always(concat_dim >= 0);
  util::MPIRootPrintStreamInfo()
    << "Concatenating tensors along dimension " << concat_dim;

  // init src1
  DataType src1_init_val = 1;
  auto src1_size = src1.get_local_real_size();
  std::vector<DataType> src1_init(src1_size, src1_init_val);
  h2::gpu::mem_copy(src1.get_buffer(), src1_init.data(), src1_size);
  // init src2
  DataType src2_init_val = 2;
  auto src2_size = src2.get_local_real_size();
  std::vector<DataType> src2_init(src2_size, src2_init_val);
  h2::gpu::mem_copy(src2.get_buffer(), src2_init.data(), src2_size);

  Concatenate(dst, src1, src2, 0);

  h2::gpu::sync();
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Concatenation done";

  using TensorProcType = Tensor<DataType, LocaleProcess, BaseAllocator>;
  auto proc_dist = Distribution::make_localized_distribution(num_dims);
  TensorProcType dst_host(LocaleProcess(), proc_dist);
  assert0(tensor::Copy(dst_host, dst, 0));

  int num_errors = 0;
  if (loc.get_rank() == 0)
  {
    for (auto it = dst_host.get_shape().index_begin();
         it != dst_host.get_shape().index_end();
         ++it)
    {
      auto idx = *it;
      auto computed = dst_host.get(idx);
      DataType ref = 0;
      if (idx[concat_dim] < src1_shape[concat_dim])
      {
        ref = src1_init_val;
      }
      else
      {
        ref = src2_init_val;
      }
      if (computed != ref)
      {
        util::MPIPrintStreamError()
          << "Error! Mismatch at " << *it << ". Computed: " << computed
          << ", ref: " << ref;
        ++num_errors;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);

  Slice(src1, src2, dst, 0);
  h2::gpu::sync();
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Split done";

  std::vector<TensorType*> src_tensors = {&src1, &src2};
  for (int i = 0; i < src_tensors.size(); ++i)
  {
    const auto& src = *src_tensors[i];
    TensorProcType host(LocaleProcess(), proc_dist);
    assert0(tensor::Copy(host, src, 0));
    int num_errors = 0;
    if (loc.get_rank() == 0)
    {
      for (auto it = host.get_shape().index_begin();
           it != host.get_shape().index_end();
           ++it)
      {
        auto idx = *it;
        auto computed = host.get(idx);
        DataType ref = i == 0 ? src1_init_val : src2_init_val;
        if (computed != ref)
        {
          util::MPIPrintStreamError()
            << "Error! Mismatch at " << *it << ". Computed: " << computed
            << ", ref: " << ref;
          ++num_errors;
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  return num_errors;
}

int main(int argc, char* argv[])
{
  h2::gpu::set_gpu(util::choose_gpu());
  MPI_Init(&argc, &argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPIPrintStreamInfo() << "Using device " << h2::gpu::current_gpu();

  using DataType = int;
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  Shape dst_shape({32, 32, 32, 5, np});
  Shape src1_shape({32, 32, 32, 2, np});
  Shape src2_shape({32, 32, 32, 3, np});

  auto overlapped_dist = Distribution::make_overlapped_distribution(
    {1, 2, 2, 1, np / 4}, {0, 1, 1, 0, 0});
  auto non_overlapped_dist =
    Distribution::make_distribution({1, 2, 2, 1, np / 4});
  assert_always((np % 4) == 0 && (np / 4 > 0));

  // concat tensors with no halo to tensor with halo
  assert0(test_concat<TensorType>(
    dst_shape, overlapped_dist, src1_shape, src2_shape, non_overlapped_dist));
  // concat tensors with no halo to tensor with halo
  assert0(test_concat<TensorType>(
    dst_shape, non_overlapped_dist, src1_shape, src2_shape, overlapped_dist));

  MPI_Barrier(MPI_COMM_WORLD);
  MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  static_cast<void>(GPU_DEVICE_RESET());

  return 0;
}
