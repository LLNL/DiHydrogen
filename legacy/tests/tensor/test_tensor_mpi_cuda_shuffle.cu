#include "distconv/distconv.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "test_tensor.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/tensor/shuffle_mpi_cuda.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_al.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/shuffle_mpi_cuda_p2p.hpp"
#include "distconv/tensor/shuffle_mpi_cuda_hybrid.hpp"
#endif // DISTCONV_HAS_P2P

#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#include <Al.hpp>

#include <iostream>
#include <cmath>

using DataType = float;

using namespace distconv;
using namespace distconv::tensor;
using namespace h2::gpu;
using AlBackend = Al::HostTransferBackend;

MPI_Comm local_comm;
int local_comm_size;

template <>
inline LocaleMPI get_locale<LocaleMPI>() {
  return LocaleMPI(MPI_COMM_WORLD);
}

template <int ND>
__global__ void init_tensor(DataType *buf,
                            Array<ND> local_shape,
                            Array<ND> halo,
                            index_t pitch,
                            Array<ND> global_shape,
                            Array<ND> global_index_base);


template <>
__global__ void init_tensor<4>(DataType *buf,
                               Array<4> local_shape,
                               Array<4> halo,
                               index_t pitch,
                               Array<4> global_shape,
                               Array<4> global_index_base) {
  auto local_real_shape = local_shape + halo * 2;
  for (index_t l = blockIdx.y; l < local_shape[3]; l += gridDim.y) {
    for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
          Array<4> local_idx = {i, j, k, l};
          size_t local_offset = get_offset(
              local_idx + halo, local_real_shape, pitch);
          auto global_idx = global_index_base + local_idx;
          size_t global_offset = get_offset(
              global_idx, global_shape);
          buf[local_offset] = global_offset;
        }
      }
    }
  }
}

template <>
__global__ void init_tensor<5>(DataType *buf,
                               Array<5> local_shape,
                               Array<5> halo,
                               index_t pitch,
                               Array<5> global_shape,
                               Array<5> global_index_base) {
  auto local_real_shape = local_shape + halo * 2;
  for (index_t m = blockIdx.y; m < local_shape[4]; m += gridDim.y) {
    for (index_t l = blockIdx.x; l < local_shape[3]; l += gridDim.x) {
      for (index_t k = 0; k < local_shape[2]; ++k) {
        for (index_t j = 0; j < local_shape[1]; ++j) {
          for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
            Array<5> local_idx = {i, j, k, l, m};
            size_t local_offset = get_offset(
                local_idx + halo, local_real_shape, pitch);
            auto global_idx = global_index_base + local_idx;
            size_t global_offset = get_offset(
                global_idx, global_shape);
            buf[local_offset] = global_offset;
          }
        }
      }
    }
  }
}

template <int ND>
__global__ void check_tensor(const DataType *buf,
                             Array<ND> local_shape,
                             Array<ND> halo,
                             index_t pitch,
                             Array<ND> global_shape,
                             const Array<ND> global_index_base,
                             int *error_counter);

template <>
__global__ void check_tensor<3>(const DataType *buf,
                                Array<3> local_shape,
                                Array<3> halo,
                                index_t pitch,
                                Array<3> global_shape,
                                const Array<3> global_index_base,
                                int *error_counter) {
  Array<3> local_real_shape = local_shape + halo * 2;
  for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
    for (index_t j = 0; j < local_shape[1]; ++j) {
      for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
        Array<3> local_idx = {i, j, k};
        size_t local_offset = get_offset(
            local_idx + halo, local_real_shape, pitch);
        Array<3> global_idx = global_index_base + local_idx;
        int global_offset = get_offset(
            global_idx, global_shape);
        auto stored = buf[local_offset];
        if (stored != global_offset) {
          atomicAdd(error_counter, 1);
          printf("Error at (%d, %d, %d); ref: %d, stored: %f, global_index_base(%d, %d, %d)\n",
                 (int)global_idx[0], (int)global_idx[1], (int)global_idx[2],
                 global_offset, stored,
                 (int)global_index_base[0], (int)global_index_base[1],
                 (int)global_index_base[2]);
        }
      }
    }
  }
}

template <>
__global__ void check_tensor<4>(const DataType *buf,
                                Array<4> local_shape,
                                Array<4> halo,
                                index_t pitch,
                                Array<4> global_shape,
                                const Array<4> global_index_base,
                                int *error_counter) {
  auto local_real_shape = local_shape + halo * 2;
  for (index_t l = blockIdx.y; l < local_shape[3]; l += gridDim.y) {
    for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
          Array<4> local_idx = {i, j, k, l};
          size_t local_offset = get_offset(
              local_idx + halo, local_real_shape, pitch);
          auto global_idx = global_index_base + local_idx;
          int global_offset = get_offset(global_idx, global_shape);
          auto stored = buf[local_offset];
          if (stored != global_offset) {
            atomicAdd(error_counter, 1);
#if 1
            printf("Error at (%d, %d, %d, %d); ref: %d, stored: %f, global_index_base(%d, %d, %d, %d), local_offset: %d\n",
                   (int)global_idx[0], (int)global_idx[1],
                   (int)global_idx[2], (int)global_idx[3],
                   global_offset, stored,
                   (int)global_index_base[0], (int)global_index_base[1],
                   (int)global_index_base[2], (int)global_index_base[3],
                   (int)local_offset);
#endif
          }
        }
      }
    }
  }
}

template <>
__global__ void check_tensor<5>(const DataType *buf,
                                Array<5> local_shape,
                                Array<5> halo,
                                index_t pitch,
                                Array<5> global_shape,
                                const Array<5> global_index_base,
                                int *error_counter) {
  auto local_real_shape = local_shape + halo * 2;
  for (index_t m = blockIdx.y; m < local_shape[4]; m += gridDim.y) {
    for (index_t l = blockIdx.x; l < local_shape[3]; l += gridDim.x) {
      for (index_t k = 0; k < local_shape[2]; ++k) {
        for (index_t j = 0; j < local_shape[1]; ++j) {
          for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
            Array<5> local_idx = {i, j, k, l, m};
            size_t local_offset = get_offset(
                local_idx + halo, local_real_shape, pitch);
            auto global_idx = global_index_base + local_idx;
            int global_offset = get_offset(global_idx, global_shape);
            auto stored = buf[local_offset];
            if (stored != global_offset) {
              atomicAdd(error_counter, 1);
#if 1
              printf("Error at (%d, %d, %d, %d, %d); ref: %d, stored: %f, global_index_base(%d, %d, %d, %d, %d), local_offset: %d\n",
                     (int)global_idx[0], (int)global_idx[1],
                     (int)global_idx[2], (int)global_idx[3],
                     (int)global_idx[4],
                     global_offset, stored,
                     (int)global_index_base[0], (int)global_index_base[1],
                     (int)global_index_base[2], (int)global_index_base[3],
                     (int)global_index_base[4],
                     (int)local_offset);
#endif
            }
          }
        }
      }
    }
  }
}

bool is_p2p_capable(const Distribution &d) {
  int num_procs = 1;
  for (int i = 0; i < d.num_dims() - 1; ++i) {
    num_procs *= d.get_locale_shape()[i];
  }
  return num_procs <= local_comm_size;
}

template <int ND, typename TensorSrc,
          typename TensorDest>
int test_copy_shuffle(const Shape &shape,
                      const Distribution &dist_src,
                      const Distribution &dist_dest,
                      ShuffleMethod method) {

  auto loc_src = get_locale<typename TensorSrc::locale_type>();
  auto loc_dest = get_locale<typename TensorDest::locale_type>();
  auto t_src = get_tensor<TensorSrc>(shape, loc_src, dist_src);
  auto t_dest = get_tensor<TensorDest>(shape, loc_dest, dist_dest);

  util::MPIRootPrintStreamDebug()
      << "Transposing " << t_src << " to " << t_dest;

  assert_always(t_src.allocate() == 0);
  DataType *buf = t_src.get_buffer();
  //assert_always(buf);

  dim3 block_dim(256);
  dim3 grid_dim(shape[-2], shape[-1]);
  init_tensor<ND><<<grid_dim, block_dim>>>(
      buf,
      t_src.get_local_shape(),
      t_src.get_overlap(),
      t_src.get_pitch(),
      t_src.get_shape(),
      t_src.get_global_index());

  h2::gpu::sync();

  assert_always(t_dest.allocate() == 0);

  TensorMPICUDAShuffler<DataType> *shuffler = nullptr;
#ifdef DISTCONV_HAS_P2P
  p2p::P2P *p2p_h = nullptr;
#endif // DISTCONV_HAS_P2P
  AlBackend::comm_type *al_comm = nullptr;
  DeviceStream stream = make_stream();

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Creating a shuffler";

  switch (method) {
    case ShuffleMethod::MPI:
      shuffler = new TensorMPICUDAShuffler<DataType>(t_src, t_dest);
      break;
    case ShuffleMethod::AL:
      al_comm = new AlBackend::comm_type(MPI_COMM_WORLD, stream);
      shuffler = new tensor::TensorMPICUDAShufflerAL<DataType>(
          t_src, t_dest, *al_comm);
      break;
#ifdef DISTCONV_HAS_P2P
    case ShuffleMethod::P2P:
      p2p_h = new p2p::P2P(MPI_COMM_WORLD);
      shuffler = new TensorMPICUDAShufflerP2P<DataType>(
          t_src, t_dest, *p2p_h);
      break;
    case ShuffleMethod::HYBRID:
      p2p_h = new p2p::P2P(MPI_COMM_WORLD);
      al_comm = new AlBackend::comm_type(MPI_COMM_WORLD, stream);
      shuffler = new tensor::TensorMPICUDAShufflerHybrid<DataType>(
          t_src, t_dest, *p2p_h, *al_comm);
      break;
#endif // DISTCONV_HAS_P2P
    default:
      util::MPIRootPrintStreamError() << "Unknown shuffle method";
      std::abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Executing shuffle_forward";
  shuffler->shuffle_forward(t_src.get_base_ptr(),
                            t_dest.get_base_ptr());
  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamInfo() << "Checking results";

  int error_counter = 0;

  int *error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  mem_copy(error_counter_d, &error_counter);

  if (t_dest.is_split_root()) {
    check_tensor<ND><<<grid_dim, block_dim>>>(
        t_dest.get_buffer(),
        t_dest.get_local_shape(),
        t_dest.get_overlap(),
        t_dest.get_pitch(),
        t_dest.get_shape(),
        t_dest.get_global_index(),
        error_counter_d);
    mem_copy(&error_counter, error_counter_d);
    util::MPIPrintStreamDebug() << "#errors: " << error_counter;
  }

  MPI_Allreduce(MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (error_counter != 0) {
    distconv::dump_tensor(t_dest, "tensor_dest", true);
    distconv::dump_tensor(t_src, "tensor_src", true);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  assert_always(error_counter == 0);

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIPrintStreamDebug()
      << "Transposing " << t_dest << " to " << t_src;

  util::MPIRootPrintStreamInfo() << "Executing shuffle_backward";
  shuffler->shuffle_backward(t_dest.get_base_ptr(),
                             t_src.get_base_ptr());

  MPI_Barrier(MPI_COMM_WORLD);
  util::MPIRootPrintStreamDebug() << "Checking results";

  if (t_src.is_split_root()) {
    check_tensor<ND><<<grid_dim, block_dim>>>(
        t_src.get_buffer(),
        t_src.get_local_shape(),
        t_src.get_overlap(),
        t_src.get_pitch(),
        t_src.get_shape(),
        t_src.get_global_index(),
        error_counter_d);
    mem_copy(&error_counter, error_counter_d);
    if (error_counter) {
      util::MPIPrintStreamError() << "#errors: " << error_counter;
    }
  }

  assert_always(error_counter == 0);

  delete shuffler;
#ifdef DISTCONV_HAS_P2P
  if (p2p_h) delete p2p_h;
#endif // DISTCONV_HAS_P2P
  if (al_comm) delete al_comm;
  destroy(stream);

  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

Distribution get_sample_dist(const Shape &shape, int np) {
  auto last_dim = shape[get_sample_dim()];
  if (last_dim >= np) {
    return make_sample_distribution(shape.num_dims(), np);
  }
  assert0(np % last_dim);
  Shape proc_shape(shape.num_dims(), 1);
  proc_shape[get_sample_dim()] = last_dim;
  proc_shape[0] = np / last_dim;
  auto split_shape = proc_shape;
  split_shape[0] = 1;
  auto d = Distribution::make_shared_distribution(proc_shape, split_shape);
  util::MPIRootPrintStreamInfo()
      << "Using strided sample distribution: " << d;
  return d;
}

template <int ND>
int run_tests(const Shape &proc_dim,
              const Shape &shape,
              ShuffleMethod method) {
  constexpr int NSD = ND - 2;
  const auto create_spatial_overlap =
      []() {
        IntVector v(NSD, 1);
        v.push_back(0);
        v.push_back(0);
        return v;
      };

  using TensorMPI = Tensor<DataType, LocaleMPI, CUDAAllocator>;
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
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist1, dist2, method)) == 0);
    // reverse
    assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
        shape, dist2, dist1, method)) == 0);
  }

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

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from sample-distributed tensor to spatially-distributed tensor.";
    auto dist1 = get_sample_dist(shape, np);
    auto dist2 = Distribution::make_distribution(proc_dim);
    bool skip = false;
#ifdef DISTCONV_HAS_P2P
    if (method == ShuffleMethod::P2P && !is_p2p_capable(dist2)) {
      util::MPIRootPrintStreamInfo()
          << "Shuffling " << dist2
          << " not supported with P2P as it involves inter-node transfer";
    }
#endif // DISTCONV_HAS_P2P
    if (!skip) {
      util::MPIRootPrintStreamInfo() << "dist1 (" << dist1
                                     << ") to dist2 (" << dist2 << ")";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist1, dist2, method)) == 0);
      MPI_Barrier(MPI_COMM_WORLD);
      // reverse
      util::MPIRootPrintStreamInfo() << "dist2 (" << dist2
                                     << ") to dist1 (" << dist1 << ")";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist2, dist1, method)) == 0);
    }
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: copy from sample-distributed tensor to spatially-distributed tensor with halo.";
    auto dist1 = get_sample_dist(shape, np);
    auto dist2 = Distribution::make_overlapped_distribution(proc_dim, create_spatial_overlap());
    bool skip = false;
#ifdef DISTCONV_HAS_P2P
    if (method == ShuffleMethod::P2P && !is_p2p_capable(dist2)) {
      util::MPIRootPrintStreamInfo()
          << "Shuffling " << dist2
          << " not supported with P2P as it involves inter-node transfer";
    }
#endif // DISTCONV_HAS_P2P
    if (!skip) {
      util::MPIRootPrintStreamInfo() << "dist1 (" << dist1
                                     << ") to dist2 (" << dist2 << ")";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist1, dist2, method)) == 0);
      MPI_Barrier(MPI_COMM_WORLD);
      // reverse
      util::MPIRootPrintStreamInfo() << "dist2 (" << dist2
                                     << ") to dist1 (" << dist1 << ")";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist2, dist1, method)) == 0);
    }
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: shrink distribution of the spatial dimensions.";
    auto dist1 = Distribution::make_overlapped_distribution(proc_dim, create_spatial_overlap());
    auto dist2 = dist1;
    auto shrunk_split_shape = dist2.get_split_shape();
    for(int i = 0; i < NSD; i++)
      shrunk_split_shape[i] = 1;
    dist2.set_split_shape(shrunk_split_shape);
    bool skip = false;
#ifdef DISTCONV_HAS_P2P
    if (method == ShuffleMethod::P2P && !is_p2p_capable(dist1)) {
      util::MPIRootPrintStreamInfo()
          << "Shuffling " << dist1
          << " not supported with P2P as it involves inter-node transfer";
    }
#endif // DISTCONV_HAS_P2P
    if (!skip) {
      util::MPIRootPrintStreamInfo() << "dist1 to dist2";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist1, dist2, method)) == 0);
      MPI_Barrier(MPI_COMM_WORLD);
      util::MPIRootPrintStreamInfo() << "dist2 to dist1";
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist2, dist1, method)) == 0);
    }
  }

  {
    MPI_Barrier(MPI_COMM_WORLD);
    util::MPIRootPrintStreamInfo()
        << "Test: shrink distribution of the spatial dimensions from sample-parallel tensors.";
    auto split_shape = proc_dim;
    for(int i = 0; i < NSD; i++)
      split_shape[i] = 1;
    Distribution dist1(proc_dim, split_shape, create_spatial_overlap(),
                       Shape(proc_dim.num_dims(), 0));
    bool skip = false;
#ifdef DISTCONV_HAS_P2P
    if (method == ShuffleMethod::P2P && !is_p2p_capable(dist1)) {
      util::MPIRootPrintStreamInfo()
          << "Shuffling " << dist1
          << " not supported with P2P as it involves inter-node transfer";
      skip = true;
    }
#endif // DISTCONV_HAS_P2P
    if (!skip) {
      auto sample_dist = get_sample_dist(shape, np);
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, dist1, sample_dist, method)) == 0);
      MPI_Barrier(MPI_COMM_WORLD);
      assert_always((test_copy_shuffle<ND, TensorMPI, TensorMPI>(
          shape, sample_dist, dist1, method)) == 0);
    }
  }

  return 0;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_cuda_shuffle pw ph pc pn [w [h [c
  [n]]]], where pw * ph * pc * pn == N with optional h, w, c, n specifying the
  dimensions of a test tensor.
 */
int main(int argc, char *argv[]) {
  const auto pop_arg =
      [&argc, &argv] {
        const std::string arg(*argv);
        argv++;
        argc--;
        return arg;
      };

  set_gpu(util::choose_gpu());
  Al::Initialize(argc, argv);

  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                      0, MPI_INFO_NULL, &local_comm);
  MPI_Comm_size(local_comm, &local_comm_size);

  const std::string bin = pop_arg();
  const auto print_usage_and_exit =
      [bin](const std::string usage) {
        util::MPIRootPrintStreamError() << "Error! Usage: " << bin
                                        << " " << usage;
        MPI_Finalize();
        exit(1);
      };

  // Parse the number of spatial dimensions
  if (argc < 1)
    print_usage_and_exit("ND");
  const int NSD = std::stoi(pop_arg());
  if(!(NSD == 2 || NSD == 3)) {
    util::MPIRootPrintStreamError() << "Invalid number of spatial dimensions: "
                                    << NSD;
    MPI_Finalize();
    exit(1);
  }
  const int ND = NSD + 2;

  // Parse the proc shape
  std::vector<std::string> dim_names;
  if(ND == 4)
    dim_names = {"w", "h", "c", "n"};
  else
    dim_names = {"w", "h", "d", "c", "n"};
  std::transform(
      dim_names.begin(), dim_names.end(), dim_names.begin(),
      [](const std::string name) {
        return std::string("proc_") + name;
      });
  if (argc < ND)
    print_usage_and_exit("ND" + util::join_spaced_array(dim_names));
  Shape proc_dim_v;
  for(int i = 0; i < ND; i++) {
    proc_dim_v.push_back(std::stoi(pop_arg()));
  }

  // Parse the tensor shape
  Shape tensor_shape_v(ND - 2, 8);
  tensor_shape_v.push_back(2);
  tensor_shape_v.push_back(np);
  for(int i = 0; i < ND; i++)
    if (argc > 0) {
      tensor_shape_v[i] = std::stoi(pop_arg());
    }

  // Run tests
  std::vector<ShuffleMethod> methods;
  if (argc > 0) {
    std::string method_name = pop_arg();
    if (method_name == "MPI") {
      methods.push_back(ShuffleMethod::MPI);
    } else if (method_name == "AL") {
      methods.push_back(ShuffleMethod::AL);
#ifdef DISTCONV_HAS_P2P
    } else if (method_name == "P2P") {
      methods.push_back(ShuffleMethod::P2P);
    } else if (method_name == "HYBRID") {
      methods.push_back(ShuffleMethod::HYBRID);
#endif
    } else {
      util::MPIRootPrintStreamError() << "Unknown method name: "
                                      << method_name;
      MPI_Finalize();
      exit(1);
    }
  } else {
    methods = {
      ShuffleMethod::MPI, ShuffleMethod::AL,
#ifdef DISTCONV_HAS_P2P
      ShuffleMethod::P2P, ShuffleMethod::HYBRID
#endif // DISTCONV_HAS_P2P
    };
  }

  MPI_Barrier(MPI_COMM_WORLD);
  for(const auto method : methods) {
    if(ND == 4) {
      run_tests<4>(proc_dim_v, tensor_shape_v, method);
    } else {
      run_tests<5>(proc_dim_v, tensor_shape_v, method);
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  util::MPIRootPrintStreamInfo() << "Completed successfully.";

  Al::Finalize();
  return 0;
}
