#pragma once

#include "distconv/cudnn/backend.hpp"

namespace distconv {
namespace leaky_relu {

template <typename Tensor>
void forward(Tensor &input, typename Tensor::data_type negative_slope,
             Tensor &output, cudaStream_t stream);

template <typename Tensor>
void backward(Tensor &input, Tensor &d_output,
              typename Tensor::data_type negative_slope, Tensor &output,
              cudaStream_t stream);

} // namespace leaky_relu

template <>
class LeakyReLU<cudnn::BackendCUDNN> {
 public:
  LeakyReLU(cudnn::BackendCUDNN &backend): m_be(backend) {}

  ~LeakyReLU() = default;

  LeakyReLU operator=(const LeakyReLU &x) {
    assert_always(&m_be == &x.m_be);
    return *this;
  }

  // input should be const, but transform::Transform, which is used
  // in the implementation, is not polymorphic with respect to
  // constness of tensor parameters. All of tensors need to
  // be non-const.
  template <typename Tensor>
  int forward(Tensor &input,
              typename Tensor::data_type negative_slope,
              Tensor &output) {
    util::MPIPrintStreamDebug()
        << "Leaky Relu FP: "
        << input << ", " << output;
    if (input.get_local_size() == 0) {
      return 0;
    }
    leaky_relu::forward(input, negative_slope, output, m_be.get_stream());
    return 0;
  }

  template <typename Tensor>
  int backward(Tensor &input,
               Tensor &d_output,
               typename Tensor::data_type negative_slope,
               Tensor &d_input) {
    util::MPIPrintStreamDebug()
        << "Leaky Relu BP: " << d_output << ", "
        << input << ", " << d_input;
    if (d_input.get_local_size() == 0) {
      return 0;
    }
    leaky_relu::backward(input, d_output, negative_slope, d_input, m_be.get_stream());
    return 0;
  }

 protected:
  cudnn::BackendCUDNN &m_be;
};

} // namespace distconv
