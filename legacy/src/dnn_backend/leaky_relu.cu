#include "distconv/dnn_backend/leaky_relu.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_mpi.hpp"

using distconv::tensor::LocaleMPI;
using distconv::tensor::CUDAAllocator;

template <typename DataType>
using Tensor = distconv::tensor::Tensor<DataType, LocaleMPI, CUDAAllocator>;

namespace distconv {
namespace leaky_relu {

template <typename DataType>
struct ForwardFunctor {
  DataType m_negative_slope;
  ForwardFunctor(DataType negative_slope): m_negative_slope(negative_slope) {}
  __device__ void operator()(const DataType &x, DataType &y) {
    auto factor  = (x > 0) ? (DataType)1 : m_negative_slope;
    y = x * factor;
  }
};

// input should be const, but Transform is not polymorphic with
// respect to constness of tensor parameters. All of tensors need to
// be non-const.
template <typename TensorType>
void forward(TensorType& input,
             typename TensorType::data_type negative_slope,
             TensorType& output,
             h2::gpu::DeviceStream stream)
{
    using DataType = typename TensorType::data_type;
    tensor::Transform(
        input, output, ForwardFunctor<DataType>(negative_slope), stream);
    return;
}

template <typename DataType>
struct BackwardFunctor {
  DataType m_negative_slope;
  BackwardFunctor(DataType negative_slope): m_negative_slope(negative_slope) {}
  __device__ void operator()(const DataType &x, const DataType &y,
                             DataType &dx) {
    auto factor  = (x > 0) ? (DataType)1 : m_negative_slope;
    dx = y * factor;
  }
};

template <typename TensorType>
void backward(TensorType& input,
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

#define INSTANTIATE_FORWARD(TYPE)                                              \
    template void forward<Tensor<TYPE>>(Tensor<TYPE> & input,                  \
                                        TYPE negative_slope,                   \
                                        Tensor<TYPE> & output,                 \
                                        h2::gpu::DeviceStream stream);
INSTANTIATE_FORWARD(float)
INSTANTIATE_FORWARD(double)
#undef INSTANTIATE_FORWARD

#define INSTANTIATE_BACKWARD(TYPE)                                             \
    template void backward<Tensor<TYPE>>(Tensor<TYPE> & input,                 \
                                         Tensor<TYPE> & d_output,              \
                                         TYPE negative_slope,                  \
                                         Tensor<TYPE> & output,                \
                                         h2::gpu::DeviceStream stream);
INSTANTIATE_BACKWARD(float)
INSTANTIATE_BACKWARD(double)
#undef INSTANTIATE_BACKWARD

} // namespace leaky_relu
} // namespace distconv
