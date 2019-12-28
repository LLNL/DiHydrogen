#pragma once

#include "distconv/cudnn/backend.hpp"
#include "distconv/tensor/algorithms.hpp"
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_mpi_cuda.hpp"
#include "distconv/tensor/allreduce_al.hpp"

#include <numeric>
#include <memory>

namespace distconv {
namespace batchnorm {

template <int ND, typename TensorType>
void channel_sums_and_sqsums(
    int num_samples,
    const TensorType &input,
    TensorType &sums,
    TensorType &sqsums,
    cudaStream_t stream);

template <int ND, typename TensorType>
void sums_to_statistics(
    index_t num_per_sum,
    typename TensorType::data_type decay,
    TensorType &global_mean,
    TensorType &global_var,
    TensorType &running_mean,
    TensorType &running_var,
    cudaStream_t stream);

template <int ND, typename TensorType>
void batch_normalization(
    int num_samples,
    const TensorType &input,
    const TensorType &mean,
    const TensorType &var,
    const TensorType &scale,
    const TensorType &bias,
    TensorType &output,
    typename TensorType::data_type epsilon,
    cudaStream_t stream);

template <int ND, typename TensorType>
void backprop1(
    int num_samples,
    const TensorType &input,
    const TensorType &d_output,
    const TensorType &mean,
    const TensorType &var,
    const TensorType &scale,
    TensorType &scale_gradient,
    TensorType &bias_gradient,
    TensorType &mean_gradient,
    TensorType &var_gradient,
    typename TensorType::data_type epsilon,
    cudaStream_t stream);

template <int ND, typename TensorType>
void backprop2(
    index_t num_samples,
    index_t num_per_sum,
    const TensorType &input,
    const TensorType &d_output,
    const TensorType &mean,
    const TensorType &var,
    const TensorType &scale,
    const TensorType &mean_gradient,
    const TensorType &var_gradient,
    TensorType &d_input,
    typename TensorType::data_type epsilon,
    cudaStream_t stream);

} // namespace batchnorm

template <int ND, typename DataType>
class BatchNormalization<cudnn::BackendCUDNN, ND, DataType> {
 public:
  BatchNormalization(cudnn::BackendCUDNN &backend,
                     DataType decay, DataType epsilon,
                     bool global_stats,
                     BatchnormImpl impl=BatchnormImpl::MPI):
      m_be(backend), m_decay(decay), m_epsilon(epsilon),
      m_global_stats(global_stats), m_impl(impl), m_allreducer(nullptr) {
    if (m_impl == BatchnormImpl::MPI) {
      m_allreducer = std::make_unique<tensor::AllreduceMPICUDA<DataType>>(
          m_be.get_comm(), m_be.get_stream());
    } else if (m_impl == BatchnormImpl::AL_NCCL) {
      m_allreducer = std::make_unique<tensor::AllreduceAlNCCL<DataType>>(
          m_be.get_al_nccl_comm());
    }
  }
#if 0
  BatchNormalization(cudnn::BackendCUDNN &backend,
                     DataType decay, DataType epsilon,
                     bool use_global_stats):
      BatchNormalization(backend, decay, epsilon,
                         tensor::Array<ND, bool>(true)) {
  }
#endif
  ~BatchNormalization() {}

  BatchNormalization operator=(
      const BatchNormalization<cudnn::BackendCUDNN, ND, DataType> &x) {
    assert_always(&m_be == &x.m_be);
    m_decay = x.m_decay;
    m_epsilon = x.m_epsilon;
    m_num_current_samples = x.m_num_current_samples;
    m_global_stats = x.m_global_stats;
    m_impl = x.impl;
    return *this;
  }

  template <typename Tensor>
  int forward_stage1(const Tensor &input, Tensor &mean, Tensor &var,
                     bool is_training) {
    set_num_samples(input.get_local_shape()[-1]);
    if (is_training) {
      channel_sums_and_sqsums(input, mean, var);
    }
    return 0;
  }

  template <typename Tensor>
  int forward_allreduce(Tensor &mean, Tensor &var, bool is_training) {
    if (!is_training || !m_global_stats) return 0;

    auto mean_ptr = mean.get_buffer();
    auto var_ptr = var.get_buffer();
    auto count = mean.get_local_pitched_size();
    assert_eq(count, var.get_local_pitched_size());

    // Combine allreduces of mean and var if possible
    if (mean_ptr + count == var_ptr) {
      // var comes immediately after mean
      m_allreducer->allreduce(mean_ptr, count * 2);
    } else if (mean_ptr == var_ptr + count) {
      // mean comes immediately after var
      m_allreducer->allreduce(var_ptr, count * 2);
    } else {
      m_allreducer->allreduce(mean_ptr, count);
      m_allreducer->allreduce(var_ptr, count);
    }

    return 0;
  }

  template <typename Tensor>
  int forward_stage2(const Tensor &input, Tensor &mean, Tensor &var,
                     Tensor &running_mean, Tensor &running_var,
                     Tensor &scale, Tensor &bias,
                     Tensor &output,
                     bool is_training) {
    if (is_training) {
      // the sample dimension of the input tensor is assumed to be
      // properly reshaped if necessary (e.g., for the last mini batch
      // in an epoch)
      auto stat_shape = m_global_stats ? input.get_shape() :
          input.get_local_shape();
      // Number of elements per channel. Note that the channel
      // dimension is assumed to be at the second to last dimension.
      index_t num_per_sum = stat_shape.get_size() / stat_shape[-2];

      // Sums to statistics
      sums_to_statistics(num_per_sum, mean, var, running_mean, running_var);
      batch_normalization(input, mean, var, scale, bias, output);
    } else {
      batch_normalization(input, running_mean, running_var, scale, bias, output);
    }

    return 0;
  }

  template <typename Tensor>
  int forward(const Tensor &input, Tensor &mean, Tensor &var,
              Tensor &running_mean, Tensor &running_var,
              Tensor &scale, Tensor &bias,
              Tensor &output,
              bool is_training) {
    util::MPIPrintStreamDebug()
        << "BatchNormalization: " << input << ", " << output;
    forward_stage1(input, mean, var, is_training);
    forward_allreduce(mean, var, is_training);
    forward_stage2(input, mean, var, running_mean, running_var, scale,
                   bias, output, is_training);
    return 0;
  }

  template <typename Tensor>
  int backward_stage1(const Tensor &input, const Tensor &d_output,
                      const Tensor &mean, const Tensor &var,
                      const Tensor &scale, Tensor &scale_gradient,
                      Tensor &bias_gradient, Tensor &mean_gradient,
                      Tensor &var_gradient) {
    util::MPIPrintStreamDebug() << "BatchNormalization BP stage 1";
    set_num_samples(input.get_local_shape()[-1]);
    backprop1(input, d_output, mean, var, scale, scale_gradient,
              bias_gradient, mean_gradient, var_gradient);
    return 0;
  }

  template <typename Tensor>
  int backward_allreduce(Tensor &scale_gradient, Tensor &bias_gradient,
                         Tensor &mean_gradient, Tensor &var_gradient) {
    if (!m_global_stats) return 0;

    auto mean_ptr = mean_gradient.get_buffer();
    auto var_ptr = var_gradient.get_buffer();
    auto count = mean_gradient.get_local_pitched_size();
    assert_eq(count, var_gradient.get_local_pitched_size());

    if (mean_ptr + count == var_ptr) {
      // var comes immediately after mean
      m_allreducer->allreduce(mean_ptr, count * 2);
    } else if (mean_ptr == var_ptr + count) {
      // mean comes immediately after var
      m_allreducer->allreduce(var_ptr, count * 2);
    } else {
      m_allreducer->allreduce(mean_ptr, count);
      m_allreducer->allreduce(var_ptr, count);
    }

    m_allreducer->allreduce(scale_gradient.get_buffer(),
                            scale_gradient.get_local_pitched_size());
    m_allreducer->allreduce(bias_gradient.get_buffer(),
                            bias_gradient.get_local_pitched_size());
    return 0;
  }

  template <typename Tensor>
  int backward_stage2(const Tensor &input, const Tensor &d_output,
                      const Tensor &mean, const Tensor &var,
                      const Tensor &scale, const Tensor &mean_gradient,
                      const Tensor &var_gradient, Tensor &d_input) {
    util::MPIPrintStreamDebug() << "BatchNormalization BP stage 2";

    auto stat_shape = m_global_stats ? input.get_shape() :
        input.get_local_shape();
    // Number of elements per channel. Note that the channel
    // dimension is assumed to be at the second to last dimension.
    index_t num_per_sum = stat_shape.get_size() / stat_shape[-2];

    backprop2(num_per_sum, input, d_output, mean, var, scale, mean_gradient,
              var_gradient, d_input);
    return 0;
  }

  template <typename Tensor>
  int backward(const Tensor &input, const Tensor &d_output,
               const Tensor &mean, const Tensor &var,
               const Tensor &scale,
               Tensor &scale_gradient, Tensor &bias_gradient,
               Tensor &mean_gradient, Tensor &var_gradient,
               Tensor &d_input) {
    backward_stage1(input, d_output, mean, var, scale, scale_gradient,
                    bias_gradient, mean_gradient, var_gradient);
    backward_allreduce(scale_gradient, bias_gradient,
                       mean_gradient, var_gradient);
    backward_stage2(input, d_output, mean, var, scale,
                    mean_gradient, var_gradient, d_input);
    return 0;
  }

  // n: the number of the current local minibatch samples
  void set_num_samples(int n) {
    if (n != m_num_current_samples) {
      util::MPIPrintStreamDebug() << "Changing number of samples to " << n;
    }
    m_num_current_samples = n;
  }

 protected:
  cudnn::BackendCUDNN &m_be;
  DataType m_decay;
  DataType m_epsilon;
  int m_num_current_samples = 0;
  bool m_global_stats;
  BatchnormImpl m_impl;
  std::unique_ptr<tensor::Allreduce<DataType>> m_allreducer;

  template <typename Tensor>
  void channel_sums_and_sqsums(const Tensor &input, Tensor &mean,
                               Tensor &var) {
    batchnorm::channel_sums_and_sqsums<ND, Tensor>(
        m_num_current_samples, input, mean, var, m_be.get_stream());
  }

  template <typename Tensor>
  void sums_to_statistics(index_t num_per_sum, Tensor &mean,
                          Tensor &var, Tensor &running_mean,
                          Tensor &running_var) {
    batchnorm::sums_to_statistics<ND, Tensor>(num_per_sum, m_decay, mean, var,
                                              running_mean, running_var,
                                              m_be.get_stream());
  }

  template <typename Tensor>
  void batch_normalization(const Tensor &input, const Tensor &mean,
                           const Tensor &var, const Tensor &scale,
                           const Tensor &bias, Tensor &output) {
    batchnorm::batch_normalization<ND, Tensor>(m_num_current_samples,
                                               input, mean, var, scale, bias, output,
                                               m_epsilon, m_be.get_stream());
  }

  template <typename Tensor>
  void backprop1(const Tensor &input, const Tensor &d_output,
                 const Tensor &mean, const Tensor &var,
                 const Tensor &scale, Tensor &scale_gradient,
                 Tensor &bias_gradient, Tensor &mean_gradient,
                 Tensor &var_gradient) {
    batchnorm::backprop1<ND, Tensor>(
        m_num_current_samples, input, d_output, mean, var, scale,
        scale_gradient, bias_gradient, mean_gradient, var_gradient,
        m_epsilon, m_be.get_stream());
  }

  template <typename Tensor>
  void backprop2(index_t num_per_sum, const Tensor &input,
                 const Tensor &d_output, const Tensor &mean,
                 const Tensor &var, const Tensor &scale,
                 const Tensor &mean_gradient, const Tensor &var_gradient,
                 Tensor &d_input) {
    batchnorm::backprop2<ND, Tensor>(
        m_num_current_samples, num_per_sum, input, d_output, mean, var, scale,
        mean_gradient, var_gradient, d_input, m_epsilon, m_be.get_stream());
  }
};

} // namespace distconv
