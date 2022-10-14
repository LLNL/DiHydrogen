#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"
#include "test_tensor.hpp"

#include <iostream>
#include <vector>

using namespace distconv;
using namespace distconv::tensor;
using namespace distconv::util;

template <>
inline LocaleMPI get_locale() {
  LocaleMPI loc(MPI_COMM_WORLD);
  return loc;
}

template <typename DataType>
__global__ void init_tensor(DataType *buf,
                            Array<3> local_shape,
                            Array<3> halo,
                            index_t pitch,
                            Array<3> global_shape,
                            Array<3> global_index_base) {
  Array<3> local_real_shape = local_shape + halo * 2;
  for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
    for (index_t j = 0; j < local_shape[1]; ++j) {
      for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
        Array<3> local_idx = {i, j, k};
        size_t local_offset = get_offset(
            local_idx + halo, local_real_shape, pitch);
        Array<3> global_idx = global_index_base + local_idx;
        size_t global_offset = get_offset(
            global_idx, global_shape);
        buf[local_offset] = global_offset;
      }
    }
  }
}

template <typename DataType, typename UnaryFunction>
__global__ void check_tensor(const DataType *buf,
                             Array<3> local_shape,
                             Array<3> halo,
                             index_t pitch,
                             Array<3> global_shape,
                             Array<3> global_index_base,
                             UnaryFunction op,
                             int *error_counter) {
  Array<3> local_real_shape = local_shape + halo * 2;
  for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
    for (index_t j = 0; j < local_shape[1]; ++j) {
      for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
        Array<3> local_idx = {i, j, k};
        size_t local_offset = get_offset(
            local_idx + halo, local_real_shape, pitch);
        Array<3> global_idx = global_index_base + local_idx;
        int global_offset = get_offset(global_idx, global_shape);
        DataType ref = op(global_offset);
        DataType stored = buf[local_offset];
        if (stored != ref) {
          atomicAdd(error_counter, 1);
          printf("Error at (%lu, %lu, %lu)@(%lu, %lu, %lu); ref: %d, stored: %d\n",
                 global_idx[0], global_idx[1], global_idx[2],
                 i, j, k, (int)ref, (int)stored);
        }
      }
    }
  }
}


template <typename DataType>
struct times2_functor {
  __device__ DataType operator()(const DataType x) const {
    return x * 2;
  }
};

template <typename DataType>
struct copy_functor {
  __device__ void operator()(const DataType &x, DataType &y) const {
    y = x;
  }
};

template <typename TensorType>
inline int test_transform(const Shape &shape,
                          const Distribution &dist) {
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  TensorType t2 = get_tensor<TensorType>(shape, loc, dist);

  MPIRootPrintStreamInfo() << "Shape: " << t.get_shape();
  MPIPrintStreamInfo() << "Local real shape: " << t.get_local_real_shape();
  MPIRootPrintStreamInfo() << "Distribution: " << t.get_distribution();

  assert0(t.allocate());
  assert0(t2.allocate());

  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();
  size_t pitch = t.get_pitch();
  MPIPrintStreamDebug() << "Base offset: " << base_offset
                        << ", global offset: " << t.get_global_index()
                        << ", pitch: " << pitch;

  init_tensor<<<4, 4>>>(buf,
                        t.get_local_shape(),
                        dist.get_overlap(),
                        t.get_pitch(),
                        t.get_shape(),
                        t.get_global_index());
  h2::gpu::sync();

  auto times2_lambda = [] __device__ (int &x) {
    x = x * 2;
  };

  Transform(t, times2_lambda);

  int error_counter = 0;
  int *error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  h2::gpu::mem_copy(error_counter_d, &error_counter);
  check_tensor<<<1, 1>>>(buf,
                         t.get_local_shape(),
                         dist.get_overlap(),
                         t.get_pitch(),
                         t.get_shape(),
                         t.get_global_index(),
                         times2_functor<typename TensorType::data_type>(),
                         error_counter_d);

  h2::gpu::mem_copy(&error_counter, error_counter_d);
  assert0(error_counter);
  h2::gpu::sync();
  MPI_Barrier(MPI_COMM_WORLD);
  MPIRootPrintStreamInfo() << "Transform with times2_lambda completed.";

  Transform(t, t2, copy_functor<typename TensorType::data_type>());
  check_tensor<<<1, 1>>>(t2.get_buffer(),
                         t2.get_local_shape(),
                         dist.get_overlap(),
                         t2.get_pitch(),
                         t2.get_shape(),
                         t2.get_global_index(),
                         times2_functor<typename TensorType::data_type>(),
                         error_counter_d);
  h2::gpu::mem_copy(&error_counter, error_counter_d);
  assert0(error_counter);

  using TensorTypeLong = Tensor<long, typename TensorType::locale_type,
                                typename TensorType::allocator_type>;
  TensorTypeLong t3 = get_tensor<TensorTypeLong>(shape, loc, dist);
  assert0(t3.allocate());
  Transform(t2, t3, [] __device__ (typename TensorType::data_type &x,
                                   typename TensorTypeLong::data_type &y) {
                      y = static_cast<typename TensorTypeLong::data_type>(x);
                    });
  check_tensor<<<1, 1>>>(t3.get_buffer(),
                         t3.get_local_shape(),
                         dist.get_overlap(),
                         t3.get_pitch(),
                         t3.get_shape(),
                         t3.get_global_index(),
                         times2_functor<typename TensorTypeLong::data_type>(),
                         error_counter_d);
  h2::gpu::mem_copy(&error_counter, error_counter_d);
  assert0(error_counter);

  h2::gpu::sync();
  MPI_Barrier(MPI_COMM_WORLD);
  MPIRootPrintStreamInfo() << "Transform with copy_functor completed.";

  return 0;
}

template <int ND, typename TensorType>
inline int test_reduce(const Shape &shape,
                       const Shape &reduce_shape,
                       const Distribution &dist) {
  const int num_dims = shape.num_dims();
  using DataType = typename TensorType::data_type;
  using LocaleType = typename TensorType::locale_type;

  for (int i = 0; i < num_dims; ++i) {
    assert_always(shape[i] == reduce_shape[i] ||
                  reduce_shape[i] == 1);
  }

  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  util::MPIPrintStreamDebug() << "Shape: " << t.get_shape()
                              << ", local real shape: " << t.get_local_real_shape()
                              << ", distribution: " << t.get_distribution();
  assert0(t.allocate());

  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();
  size_t pitch = t.get_pitch();
  util::MPIPrintStreamDebug() << "Base offset: " << base_offset
                              << ", global offset: " << t.get_global_index()
                              << ", pitch: " << pitch;

  init_tensor<<<4, 4>>>(buf,
                        t.get_local_shape(),
                        dist.get_overlap(),
                        t.get_pitch(),
                        t.get_shape(),
                        t.get_global_index());
  h2::gpu::sync();

  auto reduce_dist = dist;
  auto reduce_split_shape = reduce_dist.get_split_shape();
  for (int i = 0; i < num_dims; ++i) {
    if (reduce_shape[i] == 1) {
      reduce_split_shape[i] = 1;
    }
  }
  reduce_dist.set_split_shape(reduce_split_shape);
  TensorType reduce_tensor = get_tensor<TensorType>(
      reduce_shape, loc, reduce_dist);
  assert0(reduce_tensor.allocate());
  reduce_tensor.zero();
  util::MPIPrintStreamDebug() << "Reduction tensor: " << reduce_tensor;

  ReduceSum<ND, DataType, LocaleType, typename TensorType::allocator_type>(
      t, t.get_local_shape(), reduce_tensor);

  h2::gpu::sync();
  util::MPIPrintStreamDebug() << "Reduction done";

  using TensorProcType = Tensor<DataType, tensor::LocaleProcess,
                                tensor::BaseAllocator>;
  auto proc_dist = Distribution::make_localized_distribution(num_dims);
  TensorProcType input_tensor_host(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(input_tensor_host, t, 0));
  TensorProcType reduce_tensor_host(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(reduce_tensor_host, reduce_tensor, 0));
  int num_errors = 0;
  if (loc.get_rank() == 0) {
    Shape reduced_space(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      if (reduce_shape[i] == 1) {
        reduced_space[i] = shape[i];
      } else {
        reduced_space[i] = 1;
      }
    }
    for (auto it = reduce_tensor_host.get_shape().index_begin();
         it != reduce_tensor_host.get_shape().index_end(); ++it) {
      DataType computed = reduce_tensor_host.get(*it);
      DataType ref = 0;
      for (auto input_it = reduced_space.index_begin();
           input_it != reduced_space.index_end(); ++input_it) {
        ref += input_tensor_host.get(*input_it + *it);
      }
      if (computed != ref) {
        util::MPIPrintStreamError() << "Error! Mismatch at " << *it
                                    << ". Computed: " << computed
                                    << ", ref: " << ref;
        ++num_errors;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (num_errors > 0) return num_errors;

  // Nested reduction
  reduce_dist.set_split_shape(Shape(num_dims, 1));
  TensorType single_tensor = get_tensor<TensorType>(
      Shape(num_dims, 1), loc, reduce_dist);
  assert0(single_tensor.allocate());
  single_tensor.zero();
  util::MPIRootPrintStreamDebug()
      << "Single-element tensor: " << single_tensor;
  ReduceSum<ND, DataType, LocaleType, typename TensorType::allocator_type>(
      reduce_tensor, reduce_tensor.get_local_shape(), single_tensor);
  TensorProcType single_tensor_host(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(single_tensor_host, single_tensor, 0));
  if (loc.get_rank() == 0) {
    DataType ref = (shape.get_size() - 1) * shape.get_size() / 2;
    DataType computed = single_tensor_host.get(IndexVector(num_dims, 0));
    if (ref != computed) {
      util::MPIPrintStreamError()
          << "Error! Mismatch at the final reduction to a single value. "
          << "Computed: " << computed << ", ref: " << ref;
      ++num_errors;
    }
  }
  return num_errors;
}

template <int ND, typename TensorType, typename UnaryFunction>
inline int test_transform_reduce(const Shape &shape,
                                 const Shape &reduce_shape,
                                 const Distribution &dist,
                                 const UnaryFunction &op) {
  const int num_dims = shape.num_dims();
  using DataType = typename TensorType::data_type;
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);

  assert0(t.allocate());

  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();
  size_t pitch = t.get_pitch();

  init_tensor<<<4, 4>>>(buf,
                        t.get_local_shape(),
                        dist.get_overlap(),
                        t.get_pitch(),
                        t.get_shape(),
                        t.get_global_index());
  h2::gpu::sync();

  auto reduce_dist = dist;
  auto reduce_split_shape = reduce_dist.get_split_shape();
  for (int i = 0; i < num_dims; ++i) {
    if (reduce_shape[i] == 1) {
      reduce_split_shape[i] = 1;
    }
  }
  reduce_dist.set_split_shape(reduce_split_shape);
  TensorType reduce_tensor = get_tensor<TensorType>(
      reduce_shape, loc, reduce_dist);
  assert0(reduce_tensor.allocate());
  reduce_tensor.zero();

  TransformReduceSum<ND, DataType, LocaleType, typename TensorType::allocator_type>(
      t, reduce_tensor, op);

  h2::gpu::sync();
  util::MPIPrintStreamDebug() << "Reduction done";

  using TensorProcType = Tensor< DataType, LocaleProcess,
                                BaseAllocator>;
#if 0
  // Overload of Copy fails
  TensorProcType input_tensor_host(LocaleProcess(),
                                   Distribution<num_dims>());
  assert0(tensor::Copy(input_tensor_host, t, 0));
#else
  auto proc_dist = Distribution::make_localized_distribution(num_dims);
  TensorProcType  input_tensor_host(LocaleProcess(),
                                    proc_dist);
  assert0(tensor::Copy(input_tensor_host, t, 0));
#endif
  TensorProcType reduce_tensor_host(LocaleProcess(),
                                    proc_dist);
  assert0(tensor::Copy(reduce_tensor_host, reduce_tensor, 0));
  int num_errors = 0;
  if (loc.get_rank() == 0) {
    Shape reduced_space(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      if (reduce_shape[i] == 1) {
        reduced_space[i] = shape[i];
      } else {
        reduced_space[i] = 1;
      }
    }
    for (auto it = reduce_tensor_host.get_shape().index_begin();
         it != reduce_tensor_host.get_shape().index_end(); ++it) {
      DataType computed = reduce_tensor_host.get(*it);
      DataType ref = 0;
      for (auto input_it = reduced_space.index_begin();
           input_it != reduced_space.index_end(); ++input_it) {
        ref += op(input_tensor_host.get(*it + *input_it));
      }
      if (computed != ref) {
        util::MPIPrintStreamError() << "Error! Mismatch at " << *it
                                    << ". Computed: " << computed
                                    << ", ref: " << ref;
        ++num_errors;
      }
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return num_errors;
}

template <int ND, typename TensorType, typename UnaryFunction1,
          typename UnaryFunction2>
inline int test_transform_reduce2(const Shape &shape,
                                  const Shape &reduce_shape1,
                                  const Shape &reduce_shape2,
                                  const Distribution &dist,
                                  const UnaryFunction1 &op1,
                                  const UnaryFunction2 &op2) {
  const int num_dims = shape.num_dims();
  using DataType = typename TensorType::data_type;
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);

  assert0(t.allocate());

  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();
  size_t pitch = t.get_pitch();

  init_tensor<<<4, 4>>>(buf,
                        t.get_local_shape(),
                        dist.get_overlap(),
                        t.get_pitch(),
                        t.get_shape(),
                        t.get_global_index());
  h2::gpu::sync();

  auto reduce_dist1 = dist;
  auto reduce_split_shape1 = reduce_dist1.get_split_shape();
  for (int i = 0; i < num_dims; ++i) {
    if (reduce_shape1[i] == 1) {
      reduce_split_shape1[i] = 1;
    }
  }
  reduce_dist1.set_split_shape(reduce_split_shape1);
  TensorType reduce_tensor1 = get_tensor<TensorType>(
      reduce_shape1, loc, reduce_dist1);
  assert0(reduce_tensor1.allocate());
  reduce_tensor1.zero();

  auto reduce_dist2 = dist;
  auto reduce_split_shape2 = reduce_dist2.get_split_shape();
  for (int i = 0; i < num_dims; ++i) {
    if (reduce_shape2[i] == 1) {
      reduce_split_shape2[i] = 1;
    }
  }
  reduce_dist2.set_split_shape(reduce_split_shape2);
  TensorType reduce_tensor2 = get_tensor<TensorType>(
      reduce_shape2, loc, reduce_dist2);
  assert0(reduce_tensor2.allocate());
  reduce_tensor2.zero();

  TransformReduceSum<ND, DataType, LocaleType, typename TensorType::allocator_type>(
      t, reduce_tensor1, op1, reduce_tensor2, op2);

  h2::gpu::sync();
  util::MPIPrintStreamDebug() << "Reduction done";

  using TensorProcType = Tensor<DataType,
                                tensor::LocaleProcess,
                                tensor::BaseAllocator>;

  auto proc_dist = Distribution::make_localized_distribution(num_dims);
  TensorProcType input_tensor_host(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(input_tensor_host, t, 0));
  TensorProcType reduce_tensor_host1(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(reduce_tensor_host1, reduce_tensor1, 0));
  int num_errors = 0;
  if (loc.get_rank() == 0) {
    Shape reduced_space(reduce_shape1);
    for (int i = 0; i < num_dims; ++i) {
      if (reduced_space[i] == 1) {
        reduced_space[i] = shape[i];
      } else {
        reduced_space[i] = 1;
      }
    }
    for (auto it = reduce_tensor_host1.get_shape().index_begin();
         it != reduce_tensor_host1.get_shape().index_end(); ++it) {
      DataType computed = reduce_tensor_host1.get(*it);
      DataType ref = 0;
      for (auto input_it = reduced_space.index_begin();
           input_it != reduced_space.index_end(); ++input_it) {
        ref += op1(input_tensor_host.get(*input_it + *it));
      }
      if (computed != ref) {
        util::MPIPrintStreamError() << "Error! Mismatch at " << *it
                                    << ". Computed: " << computed
                                    << ", ref: " << ref;
        ++num_errors;
      }
    }
  }

  TensorProcType reduce_tensor_host2(tensor::LocaleProcess(), proc_dist);
  assert0(tensor::Copy(reduce_tensor_host2, reduce_tensor2, 0));
  if (loc.get_rank() == 0) {
    Shape reduced_space(reduce_shape2);
    for (int i = 0; i < num_dims; ++i) {
      if (reduced_space[i] == 1) {
        reduced_space[i] = shape[i];
      } else {
        reduced_space[i] = 1;
      }
    }
    for (auto it = reduce_tensor_host2.get_shape().index_begin();
         it != reduce_tensor_host2.get_shape().index_end(); ++it) {
      DataType computed = reduce_tensor_host2.get(*it);
      DataType ref = 0;
      for (auto input_it = reduced_space.index_begin();
           input_it != reduced_space.index_end(); ++input_it) {
        ref += op2(input_tensor_host.get(*input_it + *it));
      }
      if (computed != ref) {
        util::MPIPrintStreamError() << "Error! Mismatch at " << *it
                                    << ". Computed: " << computed
                                    << ", ref: " << ref;
        ++num_errors;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return num_errors;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_cuda_algorithms, where N must
  be >= 4 and divisible by 4.
 */
int main(int argc, char *argv[]) {
    h2::gpu::set_gpu(util::choose_gpu());
    MPI_Init(&argc, &argv);
    int pid;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    MPIPrintStreamInfo() << "Using device " << h2::gpu::current_gpu();

    constexpr int num_dims = 3;
    using DataType = int;

    using TensorMPI = Tensor<DataType, LocaleMPI, CUDAAllocator>;
    auto dist =
        Distribution::make_overlapped_distribution({2, 2, np / 4}, {1, 1, 0});
    assert_always((np % 4) == 0 && (np / 4 > 0));
    Shape tensor_shape({4, 4, 4});

    {
        assert0(test_transform<TensorMPI>(tensor_shape, dist));
        MPIRootPrintStreamInfo() << "test_transform success";
  }

  Shape reduced_dim(num_dims, 2);
  for (auto it = reduced_dim.index_begin();
       it != reduced_dim.index_end(); ++it) {
    auto reduce_shape = tensor_shape;
    for (int i = 0; i < num_dims; ++i) {
      if ((*it)[i] == 1) reduce_shape[i] = 1;
    }
    MPIRootPrintStreamInfo() << "test_reduce to " << reduce_shape;
    assert0(test_reduce<num_dims, TensorMPI>(tensor_shape, reduce_shape, dist));
  }


  for (auto it = reduced_dim.index_begin();
       it != reduced_dim.index_end(); ++it) {
    auto reduce_shape = tensor_shape;
    for (int i = 0; i < num_dims; ++i) {
      if ((*it)[i] == 1) reduce_shape[i] = 1;
    }
    MPIRootPrintStreamInfo() << "test_transform_reduce to " << reduce_shape;
    assert0(test_transform_reduce<num_dims, TensorMPI>(
        tensor_shape, reduce_shape, dist,
        [] __host__ __device__ (int x) {return x * 2;}));
  }

  {
    auto reduce_shape1 = tensor_shape;
    reduce_shape1[1] = 1;
    reduce_shape1[2] = 1;
    auto reduce_shape2 = tensor_shape;
    reduce_shape2[0] = 1;
    assert0(test_transform_reduce2<num_dims, TensorMPI>(
        tensor_shape, reduce_shape1, reduce_shape2, dist,
        [] __host__ __device__ (int x) {return x;},
        [] __host__ __device__ (int x) {return x * x;}));
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPIRootPrintStreamInfo() << "Completed successfully.";

  MPI_Finalize();
  GPU_DEVICE_RESET();
  return 0;
}
