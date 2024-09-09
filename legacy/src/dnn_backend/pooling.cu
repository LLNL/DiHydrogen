#include "distconv/dnn_backend/pooling.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

namespace
{

// using namespace distconv;
namespace dc = distconv;
namespace tensor = dc::tensor;
namespace util = dc::util;
using index_t = dc::index_t;

template <int ND>
using Array = tensor::Array<ND>;

template <typename DataType>
using Tensor =
  tensor::Tensor<DataType, tensor::LocaleMPI, tensor::CUDAAllocator>;

template <int ND, typename DataType>
__global__ void bp_accumulate_sum_kernel(DataType* tensor,
                                         const Array<ND> tensor_shape,
                                         const Array<ND> dst_offset,
                                         const Array<ND> src_offset,
                                         const Array<ND> region)
{
  index_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= region.get_size())
    return;
  Array<ND> offset;
  for (int i = 0; i < ND; ++i)
  {
    offset[i] = idx % region[i];
    idx = idx / region[i];
  }
  DataType src = tensor[tensor::get_offset(src_offset + offset, tensor_shape)];
  DataType& dst = tensor[tensor::get_offset(dst_offset + offset, tensor_shape)];
  dst += src;
#if 0
  if (blockIdx.x == 0 && threadIdx.x < 32) {
    printf("DST: %d\n",
           (int)tensor::get_offset(dst_offset + offset, tensor_shape));
  }
#endif
}

template <int ND, typename DataType>
void bp_accumulate_sum_nd(Tensor<DataType>& tensor,
                          const dc::IndexVector& dst,
                          const dc::IndexVector& src,
                          const dc::tensor::Shape& shape)
{
  auto size = shape.get_size();
  const int bsize = 256;
  int gsize = (size + bsize - 1) / bsize;
  util::MPIPrintStreamDebug()
    << "Accumulating " << src << " to " << dst << " of sub tensor with shape "
    << shape << " of " << tensor.get_local_pitched_shape() << "\n";

#if 0
  {
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
    DataType *h = nullptr;
    size_t s = tensor.get_local_real_size() * sizeof(DataType);
    cudaMallocHost(&h, s);
    DISTCONV_CHECK_CUDA(cudaMemcpy(h, tensor.get_buffer(), s,
                                   cudaMemcpyDeviceToHost));
    std::ofstream out;
    out.open("before.out",
             std::ios::out | std::ios::trunc | std::ios::binary);
    out.write((char *)h, s);
    out.close();
    DISTCONV_CHECK_CUDA(cudaFreeHost(h));
  }
#endif

  bp_accumulate_sum_kernel<ND><<<gsize, bsize>>>(
    tensor.get_buffer(), tensor.get_local_pitched_shape(), dst, src, shape);
#if 0
  {
    DataType *h = nullptr;
    size_t s = tensor.get_local_real_size() * sizeof(DataType);
    cudaMallocHost(&h, s);
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
    DISTCONV_CHECK_CUDA(cudaMemcpy(h, tensor.get_buffer(), s,
                                   cudaMemcpyDeviceToHost));
    std::ofstream out;
    out.open("after.out",
             std::ios::out | std::ios::trunc | std::ios::binary);
    out.write((char *)h, s);
    out.close();
    DISTCONV_CHECK_CUDA(cudaFreeHost(h));
  }
#endif
}

}  // namespace

namespace distconv
{

template <typename DataType>
void Pooling<BackendDNNLib, DataType>::bp_accumulate_sum(
  Tensor<DataType>& tensor,
  IndexVector const& dst,
  IndexVector const& src,
  tensor::Shape const& shape)
{
  switch (m_num_dims)
  {
  case 4: bp_accumulate_sum_nd<4, DataType>(tensor, dst, src, shape); break;
  case 5: bp_accumulate_sum_nd<5, DataType>(tensor, dst, src, shape); break;
  }
  return;
}

#define INSTANTIATE_BP_ACCUMULATE_SUM(TYPE)                                    \
  template void Pooling<BackendDNNLib, TYPE>::bp_accumulate_sum(               \
    Tensor<TYPE>& tensor,                                                      \
    const IndexVector& dst,                                                    \
    const IndexVector& src,                                                    \
    const tensor::Shape& shape)
INSTANTIATE_BP_ACCUMULATE_SUM(float);
INSTANTIATE_BP_ACCUMULATE_SUM(double);
#undef INSTANTIATE_BP_ACCUMULATE_SUM

}  // namespace distconv
