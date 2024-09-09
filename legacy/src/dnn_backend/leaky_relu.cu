#include "distconv/dnn_backend/leaky_relu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

using distconv::tensor::CUDAAllocator;
using distconv::tensor::LocaleMPI;

template <typename DataType>
using Tensor = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;

namespace
{

template <typename DataType>
struct ForwardFunctor
{
  DataType m_negative_slope;
  ForwardFunctor(DataType negative_slope) : m_negative_slope(negative_slope) {}
  __device__ void operator()(DataType const& x, DataType& y)
  {
    auto factor = (x > 0) ? (DataType) 1 : m_negative_slope;
    y = x * factor;
  }
};

template <typename DataType>
struct BackwardFunctor
{
  DataType m_negative_slope;
  BackwardFunctor(DataType negative_slope) : m_negative_slope(negative_slope) {}
  __device__ void operator()(DataType const& x, DataType const& y, DataType& dx)
  {
    auto factor = (x > 0) ? (DataType) 1 : m_negative_slope;
    dx = y * factor;
  }
};

}  // namespace

// input should be const, but Transform is not polymorphic with
// respect to constness of tensor parameters. All of tensors need to
// be non-const.
template <typename TensorType>
void distconv::leaky_relu::forward(
  TensorType& input,
  typename TensorType::data_type negative_slope,
  TensorType& output,
  h2::gpu::DeviceStream stream)
{
  using DataType = typename TensorType::data_type;
  tensor::Transform(
    input, output, ForwardFunctor<DataType>(negative_slope), stream);
  return;
}

template <typename TensorType>
void distconv::leaky_relu::backward(
  TensorType& input,
  TensorType& d_output,
  typename TensorType::data_type negative_slope,
  TensorType& d_input,
  h2::gpu::DeviceStream stream)
{
  using DataType = typename TensorType::data_type;
  tensor::Transform(input,
                    d_output,
                    d_input,
                    BackwardFunctor<DataType>(negative_slope),
                    stream);
  return;
}

#define INSTANTIATE_TEMPLATES(TYPE)                                            \
  template void distconv::leaky_relu::forward<Tensor<TYPE>>(                   \
    Tensor<TYPE> & input,                                                      \
    TYPE negative_slope,                                                       \
    Tensor<TYPE> & output,                                                     \
    h2::gpu::DeviceStream stream);                                             \
  template void distconv::leaky_relu::backward<Tensor<TYPE>>(                  \
    Tensor<TYPE> & input,                                                      \
    Tensor<TYPE> & d_output,                                                   \
    TYPE negative_slope,                                                       \
    Tensor<TYPE> & output,                                                     \
    h2::gpu::DeviceStream stream)
INSTANTIATE_TEMPLATES(float);
INSTANTIATE_TEMPLATES(double);
