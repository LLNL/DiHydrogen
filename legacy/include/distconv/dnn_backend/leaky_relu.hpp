#pragma once

#include "distconv/dnn_backend/backend.hpp"
#include "distconv/layers.hpp"
#include "distconv/runtime_gpu.hpp"

namespace distconv
{
namespace leaky_relu
{

template <typename Tensor>
void forward(Tensor& input,
             typename Tensor::data_type negative_slope,
             Tensor& output,
             h2::gpu::DeviceStream stream);

template <typename Tensor>
void backward(Tensor& input,
              Tensor& d_output,
              typename Tensor::data_type negative_slope,
              Tensor& output,
              h2::gpu::DeviceStream stream);

}  // namespace leaky_relu

template <>
class LeakyReLU<BackendDNNLib>
{
public:
  LeakyReLU(BackendDNNLib const& be) : m_stream(be.get_stream()) {}
  LeakyReLU(h2::gpu::DeviceStream stream) : m_stream(stream) {}

  ~LeakyReLU() = default;

  // input should be const, but transform::Transform, which is used
  // in the implementation, is not polymorphic with respect to
  // constness of tensor parameters. All of tensors need to
  // be non-const.
  template <typename Tensor>
  int forward(Tensor& input,
              typename Tensor::data_type negative_slope,
              Tensor& output)
  {
    util::MPIPrintStreamDebug() << "Leaky Relu FP: " << input << ", " << output;
    if (input.get_local_size() == 0)
    {
      return 0;
    }
    leaky_relu::forward(input, negative_slope, output, m_stream);
    return 0;
  }

  template <typename Tensor>
  int backward(Tensor& input,
               Tensor& d_output,
               typename Tensor::data_type negative_slope,
               Tensor& d_input)
  {
    util::MPIPrintStreamDebug()
      << "Leaky Relu BP: " << d_output << ", " << input << ", " << d_input;
    if (d_input.get_local_size() == 0)
    {
      return 0;
    }
    leaky_relu::backward(input, d_output, negative_slope, d_input, m_stream);
    return 0;
  }

private:
  h2::gpu::DeviceStream m_stream;
};

}  // namespace distconv
