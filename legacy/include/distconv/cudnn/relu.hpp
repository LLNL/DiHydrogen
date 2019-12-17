#pragma once

#include "distconv/cudnn/backend.hpp"

namespace distconv {

template <>
class ReLU<cudnn::BackendCUDNN> {
 public:
  ReLU(cudnn::BackendCUDNN &backend): m_be(backend) {
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateActivationDescriptor(&m_activation_d));
  }

  ~ReLU() {
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyActivationDescriptor(m_activation_d));
  }

  ReLU<cudnn::BackendCUDNN> operator=(
      const ReLU<cudnn::BackendCUDNN> &x) {
    assert_always(&m_be == &x.m_be);
    cudnn::copy_tensor_descriptor(m_input_d, x.m_input_d);
    cudnn::copy_tensor_descriptor(m_output_d, x.m_output_d);
    cudnn::copy_tensor_descriptor(m_d_input_d, x.m_d_input_d);
    cudnn::copy_tensor_descriptor(m_d_output_d, x.m_d_output_d);
    cudnn::copy_activation_descriptor(m_activation_d, x.m_activation_d);
    return *this;
  }

  template <typename Tensor, typename ConstTensor>
  void setup(const ConstTensor &input, const Tensor &output,
             const Tensor &d_input, const ConstTensor &d_output) {
    cudnn::setup_tensor_descriptor(m_input_d, input, false);
    cudnn::setup_tensor_descriptor(m_output_d, output, false);
    cudnn::setup_tensor_descriptor(m_d_input_d, d_input, false);
    cudnn::setup_tensor_descriptor(m_d_output_d, d_output, false);

    DISTCONV_CHECK_CUDNN(
        cudnnSetActivationDescriptor(m_activation_d,
                                     CUDNN_ACTIVATION_RELU,
                                     CUDNN_PROPAGATE_NAN,
                                     0.0));
  }

  template <typename Tensor>
  int forward(typename Tensor::data_type alpha,
              const Tensor &input,
              typename Tensor::data_type beta,
              Tensor &output) {
    util::MPIPrintStreamDebug()
        << "Relu FP: "
        << m_input_d << ", "
        << m_output_d
        << ", input ptr: " << input.get_const_base_ptr()
        << ", output ptr: " << output.get_base_ptr()
        << "\n";
    if (input.get_local_size() == 0) {
      return 0;
    }
    set_num_samples(input.get_local_shape()[-1]);
    DISTCONV_CHECK_CUDNN(
        cudnnActivationForward(
            m_be.get_handle(),
            m_activation_d,
            &alpha,
            m_input_d, input.get_const_base_ptr(),
            &beta,
            m_output_d, output.get_base_ptr()));
    return 0;
  }

  template <typename Tensor>
  int backward(typename Tensor::data_type alpha,
               Tensor &output,
               const Tensor &d_output,
               const Tensor &input,
               typename Tensor::data_type beta,
               Tensor &d_input) {
    util::MPIPrintStreamDebug()
        << "Relu BP: "
        << m_input_d << ", "
        << m_d_output_d << ", "
        << m_output_d << ", "
        << m_d_input_d;
    if (d_input.get_local_size() == 0) {
      return 0;
    }
    set_num_samples(d_input.get_local_shape()[-1]);
    DISTCONV_CHECK_CUDNN(
        cudnnActivationBackward(
            m_be.get_handle(),
            m_activation_d,
            &alpha,
            m_output_d, output.get_const_base_ptr(),
            m_d_output_d, d_output.get_const_base_ptr(),
            m_input_d, input.get_const_base_ptr(),
            &beta,
            m_d_input_d, d_input.get_base_ptr()));
    return 0;
  }

  void set_num_samples(int n) {
    if (n != cudnn::get_tensor_num_samples(m_input_d)) {
      cudnn::set_tensor_num_samples(m_input_d, n);
      cudnn::set_tensor_num_samples(m_output_d, n);
      cudnn::set_tensor_num_samples(m_d_input_d, n);
      cudnn::set_tensor_num_samples(m_d_output_d, n);
    }
  }

 protected:
  cudnn::BackendCUDNN &m_be;
  cudnnActivationDescriptor_t m_activation_d;
  cudnnTensorDescriptor_t m_input_d;
  cudnnTensorDescriptor_t m_output_d;
  cudnnTensorDescriptor_t m_d_input_d;
  cudnnTensorDescriptor_t m_d_output_d;
};

} // namespace distconv
