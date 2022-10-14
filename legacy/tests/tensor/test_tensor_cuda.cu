#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "test_tensor.hpp"
#include "distconv/util/util_gpu.hpp"

#include <iostream>

using namespace distconv;
using namespace distconv::tensor;

template <>
inline LocaleCUDA get_locale<LocaleCUDA>() {
  LocaleCUDA loc(0, 1, {0});
  return loc;
}

__global__ void init_tensor(int *buf, size_t base,
                            Array<3> shape) {
  for (index_t k = blockIdx.x; k < shape[2]; k += gridDim.x) {
    for (index_t j = 0; j < shape[1]; ++j) {
      for (index_t i = threadIdx.x; i < shape[0]; i += blockDim.x) {
        size_t offset = base + i + j * shape[0] + k * shape[0] * shape[1];
        buf[offset] = offset;
      }
    }
  }
}

__global__ void check_tensor(int *buf, size_t base,
                             Array<3> shape) {
  for (index_t k = blockIdx.x; k < shape[2]; k += gridDim.x) {
    for (index_t j = 0; j < shape[1]; ++j) {
      for (index_t i = threadIdx.x; i < shape[0]; i += blockDim.x) {
        size_t offset = base + i + j * shape[0] + k * shape[0] * shape[1];
        //printf("%d\n", buf[offset]);
        if (buf[offset] != offset) {
          printf("Error at (%zu, %zu, %zu); ref: %zu, stored: %u\n",
                 i, j, k, offset, buf[offset]);
        }
      }
    }
  }
}
#if 0
template <typename Tensor>
__global__ void check_tensor(Tensor t) {
  Array<3> shape = t.get_local_shape();
  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();
  size_t ldim = t.get_pitch();

  for (index_t k = 0; k < shape[2]; ++k) {
    for (index_t j = 0; j < shape[1]; ++j) {
      for (index_t i = 0; i < shape[0]; ++i) {
        Array<3> idx({k, j, i});
        int ref = base_offset + i + j * shape[0] + k * shape[0] * shape[1];
        if (buf[t.get_local_offset(idx)] == ref) {
          printf("Error at (%d, %d, %d); ref: %d, stored: %u\n",
                 i, j, k, ref, buf[t.get_local_offset(idx)]);
        }
      }
    }
  }
}
#endif

template <typename TensorType>
inline int test_data_access_cuda(
    const Shape &shape,
    const Distribution &dist) {
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  std::cout << "Shape: " << t.get_shape() << std::endl;
  std::cout << "Distribution: " << t.get_distribution() << std::endl;

  assert0(t.allocate());

  auto local_shape = t.get_local_shape();
  index_t base_offset = t.get_local_offset();
  int *buf = t.get_buffer();

  init_tensor<<<4, 4>>>(buf, base_offset, local_shape);
  check_tensor<<<1, 1>>>(buf, base_offset, local_shape);

  return 0;
}

/*
  Usage: ./test_tensor_cuda
 */
int main(int argc, char *argv[]) {
  h2::gpu::set_gpu(0);

  const int ND = 3;
  using DataType = int;
  using TensorCUDA = Tensor<DataType, LocaleCUDA, CUDAAllocator>;
  using TensorCUDAPitch = Tensor<DataType, LocaleCUDA, CUDAPitchedAllocator>;

  auto dist = Distribution::make_localized_distribution(ND);

  assert0(test_alloc<TensorCUDA>(Shape({2, 2, 2}), dist));

  assert0(test_data_access_cuda<TensorCUDA>(Shape({8, 8, 8}), dist));

  std::cout << "Using pitched memory\n";
  assert0(test_data_access_cuda<TensorCUDAPitch>(Shape({8, 8, 8}), dist));

  static_cast<void>(GPU_DEVICE_RESET());
  // yeah, yeah, nodiscard is the future or whatever.
  util::PrintStreamInfo() << "Completed successfully.";
  return 0;
}
