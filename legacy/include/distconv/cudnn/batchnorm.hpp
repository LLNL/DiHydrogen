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
    cudaStream_t stream,
    const std::vector<bool> &reduction_dims,
    bool reduce,
    std::unique_ptr<tensor::Allreduce<typename TensorType::data_type>> &allreducer);

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
  /**
     @reduction_dims dimensions over which statistics are aggregated.
   */
  BatchNormalization(cudnn::BackendCUDNN &backend,
                     DataType decay, DataType epsilon,
                     const std::vector<bool> &reduction_dims,
                     BatchnormImpl impl=BatchnormImpl::MPI):
      m_be(backend), m_decay(decay), m_epsilon(epsilon),
      m_reduction_dims(reduction_dims), m_use_local_stats(true),
      m_impl(impl), m_allreducer(nullptr) {
    for (auto b: m_reduction_dims) {
      if (b) {
        m_use_local_stats = false;
        break;
      }
    }
    if (m_impl == BatchnormImpl::MPI) {
      m_allreducer = std::make_unique<tensor::AllreduceMPICUDA<DataType>>(m_be.get_comm(),
                                                                          m_be.get_stream());
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
    m_reduction_dims = x.m_reduction_dims;
    m_impl = x.impl;
    return *this;
  }

  template <typename Tensor>
  int forward_stage1(const Tensor &input, Tensor &mean, Tensor &var,
                     bool is_training, bool reduce) {
    set_num_samples(input.get_local_shape()[-1]);
    if (is_training) {
      // Channel sums and sqsums
      channel_sums_and_sqsums(input, mean, var, reduce);
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
      tensor::Array<ND> stat_shape;
      for (int i = 0; i < ND; ++i) {
        stat_shape[i] = m_reduction_dims[i] ? input.get_shape()[i] :
            input.get_local_shape()[i];
      }
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
    forward_stage1(input, mean, var, is_training, true);
    forward_stage2(input, mean, var, running_mean, running_var, scale,
                   bias, output, is_training);
    return 0;
  }

  template <typename Tensor>
  int backward_stage1(const Tensor &input, const Tensor &d_output,
                      const Tensor &mean, const Tensor &var,
                      const Tensor &scale, Tensor &scale_gradient,
                      Tensor &bias_gradient, Tensor &mean_gradient,
                      Tensor &var_gradient,
                      bool reduce) {
    util::MPIPrintStreamDebug() << "BatchNormalization BP stage 1";

    set_num_samples(input.get_local_shape()[-1]);

    backprop1(input, d_output, mean, var, scale, scale_gradient,
              bias_gradient, mean_gradient, var_gradient);

    if (!m_use_local_stats && reduce) {
      // TODO: only global reduction is supported.
      assert_always(std::accumulate(
          m_reduction_dims.begin(), m_reduction_dims.end(), true,
          std::logical_and<bool>()));
      m_allreducer->allreduce(scale_gradient.get_buffer(),
                             scale_gradient.get_local_pitched_size());
      m_allreducer->allreduce(bias_gradient.get_buffer(),
                             bias_gradient.get_local_pitched_size());
      m_allreducer->allreduce(mean_gradient.get_buffer(),
                              mean_gradient.get_local_pitched_size());
      m_allreducer->allreduce(var_gradient.get_buffer(),
                              var_gradient.get_local_pitched_size());
    }

    return 0;
  }

  template <typename Tensor>
  int backward_stage2(const Tensor &input, const Tensor &d_output,
                      const Tensor &mean, const Tensor &var,
                      const Tensor &scale, const Tensor &mean_gradient,
                      const Tensor &var_gradient, Tensor &d_input) {
    util::MPIPrintStreamDebug() << "BatchNormalization BP stage 2";

    tensor::Array<ND> stat_shape;
    for (int i = 0; i < ND; ++i) {
      stat_shape[i] = m_reduction_dims[i] ? input.get_shape()[i] :
          input.get_local_shape()[i];
    }
    // Number of elements per channel. Note that the channel
    // dimension is assumed to be at the second to last dimension.
    index_t num_per_sum = stat_shape.get_size() / stat_shape[-2];

    backprop2(num_per_sum, input, d_output, mean, var, scale, mean_gradient,
              var_gradient, d_input);
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
  std::vector<bool> m_reduction_dims;
  bool m_use_local_stats;
  BatchnormImpl m_impl;
  std::unique_ptr<tensor::Allreduce<DataType>> m_allreducer;

  template <typename Tensor>
  void channel_sums_and_sqsums(const Tensor &input, Tensor &mean,
                               Tensor &var, bool reduce) {
    batchnorm::channel_sums_and_sqsums<ND, Tensor>(
        m_num_current_samples, input, mean, var, m_be.get_stream(),
        m_reduction_dims, reduce, m_allreducer);
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
