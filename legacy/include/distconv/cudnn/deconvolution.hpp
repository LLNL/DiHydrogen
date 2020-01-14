#pragma once

#include "distconv/cudnn/backend.hpp"

namespace distconv {

template <typename DataType>
class Deconvolution<cudnn::BackendCUDNN, DataType> {

 public:
  Deconvolution(cudnn::BackendCUDNN &backend):
      m_be(backend) {}
  ~Deconvolution() = default;

  template <typename Tensor>
  void setup(const Tensor &input,
             const Tensor &filter,
             const Tensor &output,
             const Tensor &d_input,
             const Tensor &d_filter,
             const Tensor &d_output,
             const int_vector &pads,
             const int_vector &strides,
             const int_vector &dilations,
             int num_groups,
             const std::string &fwd_algo,
             const std::string &bwd_data_algo,
             const std::string &bwd_filter_algo,
             size_t ws_size) {
  }

  template <typename Tensor>
  void setup_bias(const Tensor &bias) {
    //cudnn::setup_tensor_descriptor(m_bias_d, bias, false);
  }

  template <typename Tensor>
  void setup_bias_gradient(const Tensor &d_bias) {
    //cudnn::setup_tensor_descriptor(m_d_bias_d, d_bias, false);
  }
  template <typename Tensor>
  int forward(
      DataType alpha,
      Tensor &input,
      const Tensor &filter,
      typename Tensor::data_type beta,
      Tensor &output,
      bool skip_halo_exchange=false,
      bool skip_chanfilt_comm=false,
      bool dump_profile=false) {
    return 0;
  }

  template <typename Tensor>
  int apply_bias(
      typename Tensor::data_type alpha,
      const Tensor &bias,
      typename Tensor::data_type beta,
      Tensor &output) {
    if (output.get_local_size() == 0) return 0;

    set_num_samples(output.get_local_shape()[-1]);
    return 0;
  }

  cudnnConvolutionFwdAlgo_t get_fwd_algo() const {
    return m_fwd_algo;
  }

  cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo() const {
    return m_bwd_data_algo;
  }

  cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo() const {
    return m_bwd_filter_algo;
  }

  void wait() {
    m_be.wait();
  }

  void set_num_samples(int n) {
    assert_ne(n, 0);
    // Set all the tensor descriptors. No need to adjust MPI
    // datatypes, although MPI transfers will incur extra movement
#if 0
    if (n != cudnn::get_tensor_num_samples(m_input_d)) {
      util::MPIPrintStreamDebug()
          << "Setting #sample to " << n
          << " from " << cudnn::get_tensor_num_samples(m_input_d);
      //cudnn::set_tensor_num_samples(m_input_d, n);
      //setup_workspace_sizes();
    }
#endif
  }

 protected:
  cudnn::BackendCUDNN &m_be;
  cudnnConvolutionFwdAlgo_t m_fwd_algo;
  cudnnConvolutionBwdDataAlgo_t m_bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t m_bwd_filter_algo;

};

} // namespace distconv
