#include "distconv/distconv.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util_gpu.hpp"

#include "test_tensor.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/util/nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda_al.hpp"
#include "distconv/tensor/halo_exchange_cuda_mpi.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/halo_exchange_cuda_hybrid.hpp"
#include "distconv/tensor/halo_exchange_cuda_p2p.hpp"
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/halo_exchange_cuda_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM

#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#include <Al.hpp>

#include <iostream>
#include <vector>

using namespace distconv;
using namespace distconv::tensor;
using namespace h2::gpu;

using DataType = float;

template <>
inline LocaleMPI get_locale<LocaleMPI>()
{
  LocaleMPI loc(MPI_COMM_WORLD);
  return loc;
}

template <int ND>
__global__ void init_tensor(DataType* buf,
                            Array<ND> local_shape,
                            Array<ND> halo,
                            index_t pitch,
                            Array<ND> global_shape,
                            Array<ND> global_index_base);

template <>
__global__ void init_tensor<4>(DataType* buf,
                               Array<4> local_shape,
                               Array<4> halo,
                               index_t pitch,
                               Array<4> global_shape,
                               Array<4> global_index_base)
{
  auto local_real_shape = local_shape + halo * 2;
  for (index_t l = blockIdx.y; l < local_shape[3]; l += gridDim.y)
  {
    for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x)
    {
      for (index_t j = 0; j < local_shape[1]; ++j)
      {
        for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x)
        {
          Array<4> local_idx = {i, j, k, l};
          size_t local_offset =
            get_offset(local_idx + halo, local_real_shape, pitch);
          auto global_idx = global_index_base + local_idx;
          size_t global_offset = get_offset(global_idx, global_shape);
          buf[local_offset] = global_offset;
        }
      }
    }
  }
}

template <>
__global__ void init_tensor<5>(DataType* buf,
                               Array<5> local_shape,
                               Array<5> halo,
                               index_t pitch,
                               Array<5> global_shape,
                               Array<5> global_index_base)
{
  auto local_real_shape = local_shape + halo * 2;
  for (index_t m = blockIdx.y; m < local_shape[4]; m += gridDim.y)
  {
    for (index_t l = blockIdx.x; l < local_shape[3]; l += gridDim.x)
    {
      for (index_t k = 0; k < local_shape[2]; ++k)
      {
        for (index_t j = 0; j < local_shape[1]; ++j)
        {
          for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x)
          {
            Array<5> local_idx = {i, j, k, l, m};
            size_t local_offset =
              get_offset(local_idx + halo, local_real_shape, pitch);
            auto global_idx = global_index_base + local_idx;
            size_t global_offset = get_offset(global_idx, global_shape);
            buf[local_offset] = global_offset;
          }
        }
      }
    }
  }
}

template <int ND>
__global__ void check_tensor(const DataType* buf,
                             const Array<ND> local_shape,
                             const Array<ND> halo,
                             index_t pitch,
                             const Array<ND> global_shape,
                             const Array<ND> global_index_base,
                             int check_dim,
                             int* error_counter);

template <>
__global__ void check_tensor<4>(const DataType* buf,
                                const Array<4> local_shape,
                                const Array<4> halo,
                                index_t pitch,
                                const Array<4> global_shape,
                                const Array<4> global_index_base,
                                int dim,
                                int* error_counter)
{
  auto local_real_shape = local_shape + halo * 2;
  auto halo_shape = local_real_shape;
  halo_shape[dim] = halo[dim];
  for (index_t l = 0; l < halo_shape[3]; ++l)
  {
    for (index_t k = 0; k < halo_shape[2]; ++k)
    {
      for (index_t j = 0; j < halo_shape[1]; ++j)
      {
        for (index_t i = threadIdx.x; i < halo_shape[0]; i += blockDim.x)
        {
          for (int dir = 0; dir < 2; ++dir)
          {
            Array<4> local_idx = {i, j, k, l};
            if (dir == 1)
              local_idx[dim] += local_shape[dim] + halo[dim];
            size_t local_offset =
              get_offset(local_idx, local_real_shape, pitch);
            bool skip = false;
            for (int d = 0; d < 4; ++d)
            {
              if (global_index_base[d] + local_idx[d] < halo[d])
              {
                skip = true;
                continue;
              }
              else if (global_index_base[d] + local_idx[d] - halo[d]
                       >= global_shape[d])
              {
                skip = true;
                continue;
              }
            }
            if (skip)
              continue;
            auto global_idx = global_index_base + local_idx - halo;
            size_t global_offset = get_offset(global_idx, global_shape);
            auto stored = buf[local_offset];
            if (stored != global_offset)
            {
              atomicAdd(error_counter, 1);
#if 1
              printf("Error at (%d, %d, %d, %d), dir: %d; ref: %lu, stored: "
                     "%f, global_index_base(%d, %d, %d, %d), local_index(%d, "
                     "%d, %d, %d), local offset: %d\n",
                     (int) global_idx[0],
                     (int) global_idx[1],
                     (int) global_idx[2],
                     (int) global_idx[3],
                     dir,
                     global_offset,
                     stored,
                     (int) global_index_base[0],
                     (int) global_index_base[1],
                     (int) global_index_base[2],
                     (int) global_index_base[3],
                     (int) local_idx[0],
                     (int) local_idx[1],
                     (int) local_idx[2],
                     (int) local_idx[3],
                     (int) local_offset);
#endif
            }
          }
        }
      }
    }
  }
}

template <>
__global__ void check_tensor<5>(const DataType* buf,
                                const Array<5> local_shape,
                                const Array<5> halo,
                                index_t pitch,
                                const Array<5> global_shape,
                                const Array<5> global_index_base,
                                int dim,
                                int* error_counter)
{
  auto local_real_shape = local_shape + halo * 2;
  auto halo_shape = local_real_shape;
  halo_shape[dim] = halo[dim];
  for (index_t m = 0; m < halo_shape[4]; ++m)
  {
    for (index_t l = 0; l < halo_shape[3]; ++l)
    {
      for (index_t k = 0; k < halo_shape[2]; ++k)
      {
        for (index_t j = 0; j < halo_shape[1]; ++j)
        {
          for (index_t i = threadIdx.x; i < halo_shape[0]; i += blockDim.x)
          {
            for (int dir = 0; dir < 2; ++dir)
            {
              Array<5> local_idx = {i, j, k, l, m};
              if (dir == 1)
                local_idx[dim] += local_shape[dim] + halo[dim];
              size_t local_offset =
                get_offset(local_idx, local_real_shape, pitch);
              bool skip = false;
              for (int d = 0; d < 5; ++d)
              {
                if (global_index_base[d] + local_idx[d] < halo[d])
                {
                  skip = true;
                  continue;
                }
                else if (global_index_base[d] + local_idx[d] - halo[d]
                         >= global_shape[d])
                {
                  skip = true;
                  continue;
                }
              }
              if (skip)
                continue;
              auto global_idx = global_index_base + local_idx - halo;
              size_t global_offset = get_offset(global_idx, global_shape);
              auto stored = buf[local_offset];
              if (stored != global_offset)
              {
                atomicAdd(error_counter, 1);
#if 1
                printf("Error at (%d, %d, %d, %d, %d), dir: %d; ref: %lu, "
                       "stored: %f, global_index_base(%d, %d, %d, %d, %d), "
                       "local_index(%d, %d, %d, %d, %d), local offset: %d\n",
                       (int) global_idx[0],
                       (int) global_idx[1],
                       (int) global_idx[2],
                       (int) global_idx[3],
                       (int) global_idx[4],
                       dir,
                       global_offset,
                       stored,
                       (int) global_index_base[0],
                       (int) global_index_base[1],
                       (int) global_index_base[2],
                       (int) global_index_base[3],
                       (int) global_index_base[4],
                       (int) local_idx[0],
                       (int) local_idx[1],
                       (int) local_idx[2],
                       (int) local_idx[3],
                       (int) local_idx[4],
                       (int) local_offset);
#endif
              }
            }
          }
        }
      }
    }
  }
}

template <int ND>
__global__ void check_tensor_reverse(const DataType* buf,
                                     const Array<ND> local_shape,
                                     const Array<ND> halo,
                                     index_t pitch,
                                     const Array<ND> global_shape,
                                     const Array<ND> global_index_base,
                                     int check_dim,
                                     int* error_counter);

template <>
__global__ void check_tensor_reverse<4>(const DataType* buf,
                                        const Array<4> local_shape,
                                        const Array<4> halo,
                                        index_t pitch,
                                        const Array<4> global_shape,
                                        const Array<4> global_index_base,
                                        int dim,
                                        int* error_counter)
{
  auto local_real_shape = local_shape + halo * 2;
  auto boundary_shape = local_shape;
  boundary_shape[dim] = halo[dim];
  for (index_t l = 0; l < boundary_shape[3]; ++l)
  {
    for (index_t k = 0; k < boundary_shape[2]; ++k)
    {
      for (index_t j = 0; j < boundary_shape[1]; ++j)
      {
        for (index_t i = threadIdx.x; i < boundary_shape[0]; i += blockDim.x)
        {
          for (int dir = 0; dir < 2; ++dir)
          {
            Array<4> local_idx = {i, j, k, l};
            if (dir == 1)
              local_idx[dim] += local_shape[dim] - halo[dim];
            size_t local_offset =
              get_offset(local_idx + halo, local_real_shape, pitch);
            auto global_idx = global_index_base + local_idx;
            size_t global_offset = get_offset(global_idx, global_shape);
            auto stored = buf[local_offset];
#if 0
            printf("Stored at (%d, %d, %d, %d): %f, local offset: %d (%d, %d, %d, %d), (%d, %d, %d, %d)\n",
                   (int)global_idx[0], (int)global_idx[1],
                   (int)global_idx[2], (int)global_idx[3],
                   stored, (int)local_offset,
                   (int)local_idx[0], (int)local_idx[1],
                   (int)local_idx[2], (int)local_idx[3],
                   (int)local_real_shape[0],
                   (int)local_real_shape[1],
                   (int)local_real_shape[2],
                   (int)local_real_shape[3]);
#endif
            if ((dir == 0 && global_index_base[dim] == 0)
                || (dir == 1
                    && global_index_base[dim] + local_shape[dim]
                         == global_shape[dim]))
            {
              // printf("Skipping\n");
              continue;
            }
            auto ref = global_offset * 2;
            // Check the location is also fetched from the other size
            // of neighbor
            bool boundary_on_another_side = false;
            if ((dir == 1 && local_idx[dim] < halo[dim]
                 && global_index_base[dim] != 0)
                || (dir == 0 && local_shape[dim] - local_idx[dim] <= halo[dim]
                    && global_index_base[dim] + local_shape[dim]
                         < global_shape[dim]))
            {
              ref += global_offset;
              boundary_on_another_side = true;
            }
            // dim is either 0 or 1. Check this index is located at
            // the boundary of the other spatial dimension
            int dim2 = dim ^ 1;
            if (local_idx[dim2] < halo[dim2] && global_index_base[dim2] != 0)
            {
              ref += global_offset * 2;
              if (boundary_on_another_side)
                ref += global_offset;
            }
            if (local_shape[dim2] - local_idx[dim2] <= halo[dim2]
                && global_index_base[dim2] + local_shape[dim2]
                     < global_shape[dim2])
            {
              ref += global_offset * 2;
              if (boundary_on_another_side)
                ref += global_offset;
            }
            if (stored != ref)
            {
              atomicAdd(error_counter, 1);
#if 1
              printf("Error at (%d, %d, %d, %d); ref: %zu, stored: %f, "
                     "global_index_base(%d, %d, %d, %d), dir: %d\n",
                     (int) global_idx[0],
                     (int) global_idx[1],
                     (int) global_idx[2],
                     (int) global_idx[3],
                     ref,
                     stored,
                     (int) global_index_base[0],
                     (int) global_index_base[1],
                     (int) global_index_base[2],
                     (int) global_index_base[3],
                     dir);
#endif
            }
          }
        }
      }
    }
  }
}

template <>
__global__ void check_tensor_reverse<5>(const DataType* buf,
                                        const Array<5> local_shape,
                                        const Array<5> halo,
                                        index_t pitch,
                                        const Array<5> global_shape,
                                        const Array<5> global_index_base,
                                        int dim,
                                        int* error_counter)
{
  auto local_real_shape = local_shape + halo * 2;
  auto boundary_shape = local_shape;
  boundary_shape[dim] = halo[dim];
  for (index_t m = 0; m < boundary_shape[4]; ++m)
  {
    for (index_t l = 0; l < boundary_shape[3]; ++l)
    {
      for (index_t k = 0; k < boundary_shape[2]; ++k)
      {
        for (index_t j = 0; j < boundary_shape[1]; ++j)
        {
          for (index_t i = threadIdx.x; i < boundary_shape[0]; i += blockDim.x)
          {
            for (int dir = 0; dir < 2; ++dir)
            {
              Array<5> local_idx = {i, j, k, l, m};
              if (dir == 1)
                local_idx[dim] += local_shape[dim] - halo[dim];
              size_t local_offset =
                get_offset(local_idx + halo, local_real_shape, pitch);
              auto global_idx = global_index_base + local_idx;
              size_t global_offset = get_offset(global_idx, global_shape);
              auto stored = buf[local_offset];
#if 0
              printf("Stored at (%d, %d, %d, %d, %d): %f, local offset: %d (%d, %d, %d, %d, %d), (%d, %d, %d, %d, %d)\n",
                     (int)global_idx[0], (int)global_idx[1],
                     (int)global_idx[2], (int)global_idx[3],
                     (int)global_idx[4],
                     stored, (int)local_offset,
                     (int)local_idx[0], (int)local_idx[1],
                     (int)local_idx[2], (int)local_idx[3],
                     (int)local_idx[4],
                     (int)local_real_shape[0],
                     (int)local_real_shape[1],
                     (int)local_real_shape[2],
                     (int)local_real_shape[3],
                     (int)local_real_shape[4]);
#endif
              if ((dir == 0 && global_index_base[dim] == 0)
                  || (dir == 1
                      && global_index_base[dim] + local_shape[dim]
                           == global_shape[dim]))
              {
                // printf("Skipping\n");
                continue;
              }
              auto ref = global_offset * 2;
              // Check the location is also fetched from the other size
              // of neighbor
              bool boundary_on_another_side = false;
              if ((dir == 1 && local_idx[dim] < halo[dim]
                   && global_index_base[dim] != 0)
                  || (dir == 0 && local_shape[dim] - local_idx[dim] <= halo[dim]
                      && global_index_base[dim] + local_shape[dim]
                           < global_shape[dim]))
              {
                ref += global_offset;
                boundary_on_another_side = true;
              }
              // dim is either 0 or 1. Check this index is located at
              // the boundary of the other spatial dimension
              int dim2 = dim ^ 1;
              if (local_idx[dim2] < halo[dim2] && global_index_base[dim2] != 0)
              {
                ref += global_offset * 2;
                if (boundary_on_another_side)
                  ref += global_offset;
              }
              if (local_shape[dim2] - local_idx[dim2] <= halo[dim2]
                  && global_index_base[dim2] + local_shape[dim2]
                       < global_shape[dim2])
              {
                ref += global_offset * 2;
                if (boundary_on_another_side)
                  ref += global_offset;
              }
              if (stored != ref)
              {
                atomicAdd(error_counter, 1);
#if 1
                printf("Error at (%d, %d, %d, %d, %d); ref: %zu, stored: %f, "
                       "global_index_base(%d, %d, %d, %d, %d), dir: %d\n",
                       (int) global_idx[0],
                       (int) global_idx[1],
                       (int) global_idx[2],
                       (int) global_idx[3],
                       (int) global_idx[4],
                       ref,
                       stored,
                       (int) global_index_base[0],
                       (int) global_index_base[1],
                       (int) global_index_base[2],
                       (int) global_index_base[3],
                       (int) global_index_base[4],
                       dir);
#endif
              }
            }
          }
        }
      }
    }
  }
}

#ifdef DISTCONV_HAS_NVSHMEM
static std::vector<HaloExchangeMethod> nvshmem_methods = {
  HaloExchangeMethod::NVSHMEM,
  HaloExchangeMethod::NVSHMEM_GRAPH,
  HaloExchangeMethod::NVSHMEM_DIRECT,
  HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY};
#endif

void nvshmem_barrier(HaloExchangeMethod method)
{
#ifdef DISTCONV_HAS_NVSHMEM
  if (method == HaloExchangeMethod::NVSHMEM
      || method == HaloExchangeMethod::NVSHMEM_GRAPH
      || method == HaloExchangeMethod::NVSHMEM_DIRECT
      || method == HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY)
  {
    util::nvshmem::barrier();
  }
#endif
}

bool is_nvshmem_method_used(const std::vector<HaloExchangeMethod>& methods)
{
  bool used = false;
#ifdef DISTCONV_HAS_NVSHMEM
  for (const auto& m : nvshmem_methods)
  {
    if (std::find(methods.begin(), methods.end(), m) != std::end(methods))
    {
      used = true;
      break;
    }
  }
#endif
  return used;
}

template <int ND, typename Tensor>
int test_halo_exchange(const Array<ND>& shape,
                       const Distribution& dist,
                       HaloExchangeMethod method,
                       int pid,
                       int np)
{
  auto loc = get_locale<typename Tensor::locale_type>();
  auto tensor = get_tensor<Tensor>(shape, loc, dist);
  assert0(tensor.allocate());

  util::MPIRootPrintStreamInfo() << "Tensor: " << tensor;

  int is_too_small = 0;
  for (int i = 0; i < ND - 2; ++i)
  {
    if ((index_t) tensor.get_halo_width(i) > tensor.get_local_shape()[i])
    {
      util::MPIPrintStreamInfo() << "Local tensor is too small";
      is_too_small = 1;
      break;
    }
  }
  MPI_Allreduce(
    MPI_IN_PLACE, &is_too_small, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (is_too_small)
  {
    return 0;
  }

  DataType* buf = tensor.get_buffer();
  // assert_always(buf);

  dim3 block_dim(256);
  dim3 grid_dim(shape[-2], shape[-1]);
  init_tensor<ND><<<grid_dim, block_dim>>>(buf,
                                           tensor.get_local_shape(),
                                           dist.get_overlap(),
                                           tensor.get_pitch(),
                                           tensor.get_shape(),
                                           tensor.get_global_index());

  h2::gpu::sync();

  int_vector dims;
  for (int i = 0; i < ND - 2; ++i)
    dims.push_back(i);
  DeviceStream stream_main = make_stream();
  BoundaryAttributesV<std::shared_ptr<Al::NCCLBackend::comm_type>> comms;
  apply_to_spatial_sides(ND, [&](int i, Side side) {
    DeviceStream stream = make_stream_nonblocking();
    comms(i, side) = std::make_shared<Al::NCCLBackend::comm_type>(
      tensor.get_locale().get_comm(), stream);
  });
  for (int i = 0; i < ND - 2; ++i)
  {
    if (tensor.get_split_index()[i] % 2)
    {
      std::swap(comms(i, LHS), comms(i, RHS));
    }
  }

  HaloExchange<DataType, CUDAAllocator, Al::NCCLBackend>* halo_exc = nullptr;
#ifdef DISTCONV_HAS_P2P
  p2p::P2P p2p(MPI_COMM_WORLD);
#endif

  switch (method)
  {
  case HaloExchangeMethod::MPI:
    halo_exc =
      new HaloExchangeMPI<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeMPI created";
    break;
  case HaloExchangeMethod::AL:
    halo_exc =
      new HaloExchangeAL<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeAL created";
    break;
#ifdef DISTCONV_HAS_P2P
  case HaloExchangeMethod::P2P:
    halo_exc = new HaloExchangeP2P<DataType, CUDAAllocator, Al::NCCLBackend>(
      tensor, p2p);
    util::MPIRootPrintStreamInfo() << "HaloExchangeP2P created";
    break;
  case HaloExchangeMethod::HYBRID:
    halo_exc = new HaloExchangeHybrid<DataType, CUDAAllocator, Al::NCCLBackend>(
      tensor, p2p);
    util::MPIRootPrintStreamInfo() << "HaloExchangeHybrid created";
    break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  case HaloExchangeMethod::NVSHMEM:
    halo_exc =
      new HaloExchangeNVSHMEM<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEM created";
    break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
  case HaloExchangeMethod::NVSHMEM_GRAPH:
    halo_exc =
      new HaloExchangeNVSHMEMGraph<DataType, CUDAAllocator, Al::NCCLBackend>(
        tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMGraph created";
    break;
#endif // DISTCONV_HAS_CUDA_GRAPH
  case HaloExchangeMethod::NVSHMEM_DIRECT:
    halo_exc =
      new HaloExchangeNVSHMEMDirect<DataType, CUDAAllocator, Al::NCCLBackend>(
        tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMDirect created";
    break;
  case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
    halo_exc = new HaloExchangeNVSHMEMFusedNotify<DataType,
                                                  CUDAAllocator,
                                                  Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMFusedNotify created";
    break;
#endif // DISTCONV_HAS_NVSHMEM
  default:
    util::MPIRootPrintStreamError()
      << "Invalid halo exchange method: " << method;
    std::abort();
  }

  halo_exc->exchange(comms, stream_main, false, true, false, false);

  sync(stream_main);
  util::MPIPrintStreamInfo() << "Exchange completed";

  nvshmem_barrier(method);
  MPI_Barrier(MPI_COMM_WORLD);

  util::MPIRootPrintStreamInfo() << "Checking results";

  int error_counter = 0;
  int* error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  mem_copy(error_counter_d, &error_counter);
  for (int i = 0; i < dims.size(); ++i)
  {
    if (tensor.get_local_size() > 0)
    {
      check_tensor<ND><<<1, block_dim>>>(tensor.get_buffer(),
                                         tensor.get_local_shape(),
                                         dist.get_overlap(),
                                         tensor.get_pitch(),
                                         tensor.get_shape(),
                                         tensor.get_global_index(),
                                         dims[i],
                                         error_counter_d);
      h2::gpu::sync();
      std::fflush(stdout);
      std::fflush(stderr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    mem_copy(&error_counter, error_counter_d);
    if (error_counter != 0)
    {
      util::MPIPrintStreamError()
        << "Verification failed at dimension " << dims[i];
    }
    else
    {
      util::MPIPrintStreamInfo() << "Verified at dimension " << dims[i];
    }
    MPI_Allreduce(
      MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (error_counter != 0)
    {
      for (int j = 0; j < dims.size(); ++j)
      {
        halo_exc->dump_packed_halo(j);
      }
      return 1;
    }
  }
  util::MPIPrintStreamInfo() << "Deleting halo_exc";
  delete halo_exc;

  util::MPIPrintStreamInfo() << "test_halo_exchange done";
  return 0;
}

template <int ND, typename Tensor>
int test_halo_exchange_reverse(const Array<ND>& shape,
                               const Distribution& dist,
                               HaloExchangeMethod method,
                               int pid,
                               int np)
{
  auto loc = get_locale<typename Tensor::locale_type>();
  auto tensor = get_tensor<Tensor>(shape, loc, dist);
  assert0(tensor.allocate());

  int is_too_small = 0;
  for (int i = 0; i < ND - 2; ++i)
  {
    if ((index_t) tensor.get_halo_width(i) > tensor.get_local_shape()[i]
        && tensor.get_distribution().get_locale_shape()[i] != 1)
    {
      util::MPIPrintStreamInfo() << "Local tensor is too small\n";
      is_too_small = 1;
      break;
    }
  }
  MPI_Allreduce(
    MPI_IN_PLACE, &is_too_small, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  if (is_too_small)
  {
    return 0;
  }

  DataType* buf = tensor.get_buffer();
  // assert_always(buf);
  tensor.zero();

  dim3 block_dim(256);
  dim3 grid_dim(shape[-2], shape[-1]);
  init_tensor<ND><<<grid_dim, block_dim>>>(buf,
                                           tensor.get_local_shape(),
                                           dist.get_overlap(),
                                           tensor.get_pitch(),
                                           tensor.get_shape(),
                                           tensor.get_global_index());

  h2::gpu::sync();

  int_vector dims;
  for (int i = 0; i < ND - 2; ++i)
    dims.push_back(i);
  DeviceStream stream_main = make_stream();
  SpatialAttributes<ND, std::shared_ptr<Al::NCCLBackend::comm_type>> comms;
  apply_to_spatial_sides<ND>([&](int i, Side side) {
    DeviceStream stream = make_stream();
    comms(i, side) = std::make_shared<Al::NCCLBackend::comm_type>(
      tensor.get_locale().get_comm(), stream);
  });
  for (int i = 0; i < ND - 2; ++i)
  {
    if (tensor.get_split_index()[i] % 2)
    {
      std::swap(comms(i, LHS), comms(i, RHS));
    }
  }

  HaloExchange<DataType, CUDAAllocator, Al::NCCLBackend>* halo_exc = nullptr;
#ifdef DISTCONV_HAS_P2P
  p2p::P2P p2p(MPI_COMM_WORLD);
#endif // DISTCONV_HAS_P2P

  switch (method)
  {
  case HaloExchangeMethod::MPI:
    halo_exc =
      new HaloExchangeMPI<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeMPI created";
    break;
  case HaloExchangeMethod::AL:
    halo_exc =
      new HaloExchangeAL<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeAL created";
    break;
#ifdef DISTCONV_HAS_P2P
  case HaloExchangeMethod::P2P:
    halo_exc = new HaloExchangeP2P<DataType, CUDAAllocator, Al::NCCLBackend>(
      tensor, p2p);
    util::MPIRootPrintStreamInfo() << "HaloExchangeP2P created";
    break;
  case HaloExchangeMethod::HYBRID:
    halo_exc = new HaloExchangeHybrid<DataType, CUDAAllocator, Al::NCCLBackend>(
      tensor, p2p);
    util::MPIRootPrintStreamInfo() << "HaloExchangeHybrid created";
    break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  case HaloExchangeMethod::NVSHMEM:
    halo_exc =
      new HaloExchangeNVSHMEM<DataType, CUDAAllocator, Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEM created";
    break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
  case HaloExchangeMethod::NVSHMEM_GRAPH:
    halo_exc =
      new HaloExchangeNVSHMEMGraph<DataType, CUDAAllocator, Al::NCCLBackend>(
        tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMGraph created";
    break;
#endif // DISTCONV_HAS_CUDA_GRAPH
  case HaloExchangeMethod::NVSHMEM_DIRECT:
    halo_exc =
      new HaloExchangeNVSHMEMDirect<DataType, CUDAAllocator, Al::NCCLBackend>(
        tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMDirect created";
    break;
  case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
    halo_exc = new HaloExchangeNVSHMEMFusedNotify<DataType,
                                                  CUDAAllocator,
                                                  Al::NCCLBackend>(tensor);
    util::MPIRootPrintStreamInfo() << "HaloExchangeNVSHMEMFusedNotify created";
    break;
#endif // DISTCONV_HAS_NVSHMEM
  default:
    util::MPIRootPrintStreamError() << "Invalid halo exchange method";
    std::abort();
  }

  halo_exc->exchange(comms, stream_main, true, true, false, false);
  halo_exc->exchange(comms,
                     stream_main,
                     true,
                     true,
                     true,
                     false,
                     tensor::HaloExchangeAccumOp::SUM);

  sync(stream_main);
  nvshmem_barrier(method);
  MPI_Barrier(MPI_COMM_WORLD);

  util::MPIRootPrintStreamInfo() << "Checking results";

  int error_counter = 0;
  int* error_counter_d;
  GPU_MALLOC(&error_counter_d, sizeof(int));
  mem_copy(error_counter_d, &error_counter);

  for (int i = 0; i < dims.size(); ++i)
  {
    if (tensor.get_local_size() > 0)
    {
      check_tensor_reverse<ND><<<1, block_dim>>>(tensor.get_buffer(),
                                                 tensor.get_local_shape(),
                                                 dist.get_overlap(),
                                                 tensor.get_pitch(),
                                                 tensor.get_shape(),
                                                 tensor.get_global_index(),
                                                 dims[i],
                                                 error_counter_d);
      h2::gpu::sync();
      std::fflush(stdout);
      std::fflush(stderr);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    mem_copy(&error_counter, error_counter_d);
    if (error_counter != 0)
    {
      util::MPIPrintStreamError()
        << "Verification failed at dimension " << dims[i];
    }
    else
    {
      util::MPIPrintStreamInfo() << "Verified at dimension " << dims[i];
    }
    MPI_Allreduce(
      MPI_IN_PLACE, &error_counter, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (error_counter != 0)
    {
      halo_exc->dump_packed_halo(i);
      return 1;
    }
  }

  delete halo_exc;
  return 0;
}

template <int ND, typename Tensor>
int run_test(int pid,
             int np,
             const Array<ND>& tensor_shape,
             HaloExchangeMethod method,
             const Distribution& dist)
{
  MPI_Barrier(MPI_COMM_WORLD);
  nvshmem_barrier(method);
  if (test_halo_exchange<ND, Tensor>(tensor_shape, dist, method, pid, np))
  {
    util::MPIPrintStreamError() << "Test failed";
    GPU_DEVICE_RESET();
    abort();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  nvshmem_barrier(method);
  return 0;
}

template <int ND, typename Tensor>
int dispatch_tests(int pid,
                   int np,
                   const Shape& proc_dim,
                   const Shape& tensor_shape,
                   HaloExchangeMethod method)
{
  const auto create_spatial_overlap = [](const Shape& proc_dim,
                                         const int size) {
    IntVector v(ND - 2, size);
    v.push_back(0);
    v.push_back(0);
    return Distribution::make_overlapped_distribution(proc_dim, v);
  };

#if 1
  {
    util::MPIRootPrintStreamInfo()
      << "Test: partitioning inner-most 2 dimensions (size: 1)";
    run_test<ND, Tensor>(
      pid, np, tensor_shape, method, create_spatial_overlap(proc_dim, 1));
  }
#endif

#if 1
  {
    util::MPIRootPrintStreamInfo()
      << "Test: partitioning inner-most 2 dimensions (size: 2)";
    run_test<ND, Tensor>(
      pid, np, tensor_shape, method, create_spatial_overlap(proc_dim, 2));
  }
#endif

#if 1
  {
    util::MPIRootPrintStreamInfo()
      << "Test: reverse exchange with inner-most 2 dimensions (size: 1)";
    run_test<ND, Tensor>(
      pid, np, tensor_shape, method, create_spatial_overlap(proc_dim, 1));
  }
#endif

#if 1
  {
    util::MPIRootPrintStreamInfo()
      << "Test: reverse exchange with inner-most 2 dimensions (size: 2)";
    run_test<ND, Tensor>(
      pid, np, tensor_shape, method, create_spatial_overlap(proc_dim, 2));
  }
#endif

  return 0;
}

/*
  Usage: mpirun -np N ./test_tensor_mpi_cuda_shuffle px py [h [w [c
  [n]]]], where px * py == N with optional h, w, c, n specifying the
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

  std::vector<HaloExchangeMethod> methods = {HaloExchangeMethod::MPI,
                                             HaloExchangeMethod::AL,
#ifdef DISTCONV_HAS_P2P
                                             HaloExchangeMethod::P2P,
                                             HaloExchangeMethod::HYBRID
#endif // DISTCONV_HAS_P2P
  };

#ifdef DISTCONV_HAS_NVSHMEM
  methods.push_back(HaloExchangeMethod::NVSHMEM);
  methods.push_back(HaloExchangeMethod::NVSHMEM_GRAPH);
  methods.push_back(HaloExchangeMethod::NVSHMEM_DIRECT);
  methods.push_back(HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY);
#endif

  set_gpu(util::choose_gpu());
  Al::Initialize(argc, argv);
  int pid;
  int np;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  const std::string bin = pop_arg();
  const auto print_usage_and_exit = [bin](const std::string usage) {
    util::MPIRootPrintStreamError() << "Error! Usage: " << bin << " " << usage;
    MPI_Finalize();
    exit(1);
  };

  util::MPIPrintStreamInfo() << "Using device " << current_gpu();

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
    print_usage_and_exit("ND" + util::join_spaced_array(dim_names));
  Shape proc_dim_v;
  for (int i = 0; i < ND; i++)
    proc_dim_v.push_back(std::stoi(pop_arg()));

  // Parse the tensor shape
  Shape tensor_shape_v(ND - 2, 8);
  tensor_shape_v.push_back(2);
  tensor_shape_v.push_back(np);
  for (int i = 0; i < ND; i++)
    if (argc > 0)
      tensor_shape_v[i] = std::stoi(pop_arg());

  // Run tests
  if (argc > 0)
  {
    auto name = pop_arg();
    HaloExchangeMethod method = HaloExchangeMethod::AL;
    if (name == "MPI")
    {
      method = HaloExchangeMethod::MPI;
    }
    else if (name == "AL")
    {
      method = HaloExchangeMethod::AL;
#ifdef DISTCONV_HAS_P2P
    }
    else if (name == "P2P")
    {
      method = HaloExchangeMethod::P2P;
    }
    else if (name == "HYBRID")
    {
      method = HaloExchangeMethod::HYBRID;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
    }
    else if (name == "NVSHMEM")
    {
      method = HaloExchangeMethod::NVSHMEM;
    }
    else if (name == "NVSHMEM_GRAPH")
    {
      method = HaloExchangeMethod::NVSHMEM_GRAPH;
    }
    else if (name == "NVSHMEM_DIRECT")
    {
      method = HaloExchangeMethod::NVSHMEM_DIRECT;
    }
    else if (name == "NVSHMEM_FUSED_NOTIFY")
    {
      method = HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY;
#endif // DISTCONV_HAS_NVSHMEM
    }
    else
    {
      util::MPIRootPrintStreamError()
        << "Unknown halo exchange method name: " << name;
      MPI_Finalize();
      exit(1);
    }
    methods = {method};
  }

  assert_eq(proc_dim_v.size(), np);

#ifdef DISTCONV_HAS_NVSHMEM
  // Initialize NVSHMEM when used
  if (is_nvshmem_method_used(methods))
  {
    util::nvshmem::initialize(MPI_COMM_WORLD);
  }
#endif

  using TensorMPI = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  if (ND == 4)
  {
    for (auto m : methods)
    {
      dispatch_tests<4, TensorMPI>(pid, np, proc_dim_v, tensor_shape_v, m);
    }
  }
  else if (ND == 5)
  {
    for (auto m : methods)
    {
      dispatch_tests<5, TensorMPI>(pid, np, proc_dim_v, tensor_shape_v, m);
    }
  }

#ifdef DISTCONV_HAS_NVSHMEM
  // Finalize NVSHMEM when used
  if (is_nvshmem_method_used(methods))
  {
    util::MPIPrintStreamInfo() << "Finalizing nvshmem";
    util::nvshmem::finalize();
  }
#endif

  Al::Finalize();

  return 0;
}
