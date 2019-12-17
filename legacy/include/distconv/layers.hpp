#pragma once

#include "distconv/base.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"

namespace distconv {

template <typename Backend, int ND, typename DataType>
class Convolution {
 public:
  Convolution(Backend &backend);

  template <typename Tensor>
  void setup(const Tensor &input,
             const Tensor &filter,
             const Tensor &output,
             const Tensor &d_input,
             const Tensor &d_filter,
             const Tensor &d_output,
             int pad_h, int pad_w,
             int stride_h, int stride_w,
             const std::string &fwd_algo,
             const std::string &bwd_data_algo,
             const std::string &bwd_filter_algo);

  template <typename Tensor>
  void setup_bias(const Tensor &bias);

  template <typename Tensor>
  void setup_bias_gradient(const Tensor &bias_gradient);

  template <typename Tensor>
  int forward(
      typename Tensor::data_type alpha,
      Tensor &input,
      Tensor &filter,
      typename Tensor::data_type beta,
      Tensor &output);

  template <typename Tensor>
  void apply_bias(
      typename Tensor::data_type alpha,
      Tensor &bias,
      typename Tensor::data_type beta,
      Tensor &output);

  template <typename Tensor>
  int backward_data(
      typename Tensor::data_type alpha,
      Tensor &filter,
      Tensor &d_output,
      typename Tensor::data_type beta,
      Tensor &d_input);

  template <typename Tensor>
  int backward_filter(
      typename Tensor::data_type alpha,
      Tensor &input,
      Tensor &d_output,
      typename Tensor::data_type beta,
      Tensor &d_filter,
      bool reduce=true);

  template <typename Tensor>
  int backward_bias(
      typename Tensor::data_type alpha,
      Tensor &d_output,
      typename Tensor::data_type beta,
      Tensor &bias_gradient,
      bool reduce=true);


  // Wait for asynchronous tasks
  void wait();

  // Set the number of samples. Must be smaller than the original
  // sample size set by function setup
  void set_num_samples(int);
};

template <typename Backend, int ND, typename DataType>
class Pooling {
 public:
  Pooling(Backend &backend);

  template <typename Tensor>
  void setup();

  template <typename Tensor>
  int forward(
      typename Tensor::data_type alpha,
      Tensor &input,
      typename Tensor::data_type beta,
      Tensor &output);

  template <typename Tensor>
  int backward(
      typename Tensor::data_type alpha,
      Tensor &output,
      Tensor &d_output,
      Tensor &input,
      typename Tensor::data_type beta,
      Tensor &d_input);

};

template <typename Backend>
class ReLU {
 public:
  ReLU(Backend &backend);
};

template <typename Backend>
class LeakyReLU {
 public:
  LeakyReLU(Backend &backend);
};

template <typename Backend, int ND, typename DataType>
class BatchNormalization {
 public:
  BatchNormalization(Backend &backend, DataType decay, DataType epsilon);
};

} // namespace distconv
