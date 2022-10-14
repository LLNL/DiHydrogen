#pragma once

#include "distconv/tensor/tensor_mpi_cuda.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_gpu.hpp"
#include <iostream>

using namespace distconv;
using namespace distconv::tensor;

template <int ND>
__global__ void init_tensor(int *buf,
                            Array<ND> local_shape,
                            Array<ND> halo,
                            index_t pitch,
                            Array<ND> global_shape,
                            Array<ND> global_index_base);


template <>
__global__ void init_tensor<3>(int *buf,
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

template <>
__global__ void init_tensor<4>(int *buf,
                               Array<4> local_shape,
                               Array<4> halo,
                               index_t pitch,
                               Array<4> global_shape,
                               Array<4> global_index_base) {
  Array<4> local_real_shape = local_shape + halo * 2;
  for (index_t l = blockIdx.y; l < local_shape[3]; l += gridDim.y) {
    for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
          Array<4> local_idx = {i, j, k, l};
          size_t local_offset = get_offset(
              local_idx + halo, local_real_shape, pitch);
          Array<4> global_idx = global_index_base + local_idx;
          size_t global_offset = get_offset(
              global_idx, global_shape);
          buf[local_offset] = global_offset;
        }
      }
    }
  }
}

template <int ND>
__global__ void check_tensor(const int *buf,
                             Array<ND> local_shape,
                             Array<ND> halo,
                             index_t pitch,
                             Array<ND> global_shape,
                             const Array<ND> global_index_base,
                             int *error_counter);

template <>
__global__ void check_tensor<3>(const int *buf,
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
        int stored = buf[local_offset];
        if (stored != global_offset) {
          atomicAdd(error_counter, 1);
          printf("Error at (%d, %d, %d); ref: %d, stored: %d, global_index_base(%d, %d, %d)\n",
                 (int)global_idx[0], (int)global_idx[1], (int)global_idx[2],
                 global_offset, stored,
                 (int)global_index_base[0], (int)global_index_base[1],
                 (int)global_index_base[2]);
#if 0
        } else {
          printf("Correct at (%d, %d, %d); ref: %d, stored: %d, global_index_base(%d, %d, %d)\n",
                 (int)global_idx[0], (int)global_idx[1], (int)global_idx[2],
                 global_offset, stored,
                 (int)global_index_base[0], (int)global_index_base[1],
                 (int)global_index_base[2]);
#endif
        }
      }
    }
  }
}

template <>
__global__ void check_tensor<4>(const int *buf,
                                Array<4> local_shape,
                                Array<4> halo,
                                index_t pitch,
                                Array<4> global_shape,
                                const Array<4> global_index_base,
                                int *error_counter) {
  Array<4> local_real_shape = local_shape + halo * 2;
  for (index_t l = blockIdx.y; l < local_shape[3]; l += gridDim.y) {
    for (index_t k = blockIdx.x; k < local_shape[2]; k += gridDim.x) {
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t i = threadIdx.x; i < local_shape[0]; i += blockDim.x) {
          Array<4> local_idx = {i, j, k, l};
          size_t local_offset = get_offset(
              local_idx + halo, local_real_shape, pitch);
          Array<4> global_idx = global_index_base + local_idx;
          int global_offset = get_offset(
              global_idx, global_shape);
          int stored = buf[local_offset];
          if (stored != global_offset) {
            atomicAdd(error_counter, 1);
            printf("Error at (%d, %d, %d, %d); ref: %d, stored: %d, global_index_base(%d, %d, %d, %d)\n",
                   (int)global_idx[0], (int)global_idx[1], (int)global_idx[2], (int)global_idx[3],
                   global_offset, stored,
                   (int)global_index_base[0], (int)global_index_base[1],
                   (int)global_index_base[2], (int)global_index_base[3]);
#if 0
          } else {
            printf("Correct at (%d, %d, %d); ref: %d, stored: %d, global_index_base(%d, %d, %d)\n",
                   (int)global_idx[0], (int)global_idx[1], (int)global_idx[2],
                   global_offset, stored,
                   (int)global_index_base[0], (int)global_index_base[1],
                   (int)global_index_base[2]);
#endif
          }
        }
      }
    }
  }
}
