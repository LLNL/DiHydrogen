#pragma once

#include "distconv/layers.hpp"
#include "distconv/dnn_backend/backend.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/algorithms.hpp"
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_al.hpp"
#include "distconv/tensor/allreduce_mpi_cuda.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/allreduce_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM

#include <memory>
#include <numeric>

namespace distconv
{
namespace batchnorm
{

template <typename TensorType>
void channel_sums_and_sqsums(int num_dims,
                             int num_samples,
                             const TensorType& input,
                             TensorType& sums,
                             TensorType& sqsums,
                             h2::gpu::DeviceStream stream);

template <typename TensorType>
void sums_to_statistics(index_t num_per_sum,
                        typename TensorType::data_type decay,
                        TensorType& global_mean,
                        TensorType& global_var,
                        TensorType& running_mean,
                        TensorType& running_var,
                        h2::gpu::DeviceStream stream);

template <typename TensorType>
void batch_normalization(int num_dims,
                         int num_samples,
                         const TensorType& input,
                         const TensorType& mean,
                         const TensorType& var,
                         const TensorType& scale,
                         const TensorType& bias,
                         TensorType& output,
                         typename TensorType::data_type epsilon,
                         h2::gpu::DeviceStream stream);

#ifdef DISTCONV_HAS_NVSHMEM
template <typename TensorType>
void forward_all(int num_dims,
                 const TensorType& input,
                 TensorType& mean,
                 TensorType& var,
                 TensorType& running_mean,
                 TensorType& running_var,
                 TensorType& scale,
                 TensorType& bias,
                 TensorType& output,
                 typename TensorType::data_type decay,
                 typename TensorType::data_type epsilon,
                 h2::gpu::DeviceStream stream,
                 tensor::AllreduceNVSHMEM<typename TensorType::data_type>& ar);
#endif

template <typename TensorType>
void backprop1(int num_dims,
               int num_samples,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               TensorType& scale_gradient,
               TensorType& bias_gradient,
               TensorType& mean_gradient,
               TensorType& var_gradient,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream);

template <typename TensorType>
void backprop2(int num_dims,
               index_t num_samples,
               index_t num_per_sum,
               const TensorType& input,
               const TensorType& d_output,
               const TensorType& mean,
               const TensorType& var,
               const TensorType& scale,
               const TensorType& mean_gradient,
               const TensorType& var_gradient,
               TensorType& d_input,
               typename TensorType::data_type epsilon,
               h2::gpu::DeviceStream stream);

} // namespace batchnorm

template <typename DataType>
class BatchNormalization<DNNBackend<GPUDNNBackend>, DataType>
{
public:
    BatchNormalization(DNNBackend<GPUDNNBackend>& backend,
                       int num_dims,
                       DataType decay,
                       DataType epsilon,
                       bool global_stats,
                       BatchnormImpl impl = BatchnormImpl::MPI)
        : m_stream(backend.get_stream()),
          m_num_dims(num_dims),
          m_num_current_samples(0),
          m_decay(decay),
          m_epsilon(epsilon),
          m_allreducer(nullptr),
          m_impl(impl),
          m_global_stats(global_stats)
    {
        if (m_impl == BatchnormImpl::MPI)
        {
            m_allreducer = std::make_unique<tensor::AllreduceMPICUDA<DataType>>(
                backend.get_comm(), m_stream);
        }
        else if (m_impl == BatchnormImpl::AL_NCCL)
        {
            m_allreducer = std::make_unique<tensor::AllreduceAlNCCL<DataType>>(
                backend.get_al_nccl_comm());
#ifdef DISTCONV_HAS_NVSHMEM
        }
        else if (m_impl == BatchnormImpl::NVSHMEM_NATIVE)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream, tensor::AllreduceNVSHMEM<DataType>::NATIVE);
        }
        else if (m_impl == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_HOST)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream,
                tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_HOST);
        }
        else if (m_impl == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream,
                tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING);
        }
        else if (m_impl == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BUFFERED)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream,
                tensor::AllreduceNVSHMEM<
                    DataType>::RECURSIVE_DOUBLING_BUFFERED);
        }
        else if (m_impl == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BLOCK)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream,
                tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_BLOCK);
        }
        else if (m_impl == BatchnormImpl::FUSED_NVSHMEM_RECURSIVE_DOUBLING)
        {
            m_allreducer = std::make_unique<tensor::AllreduceNVSHMEM<DataType>>(
                m_stream,
                tensor::AllreduceNVSHMEM<DataType>::RECURSIVE_DOUBLING_BLOCK);
#endif // DISTCONV_HAS_NVSHMEM
        }
    }

    ~BatchNormalization() = default;

    template <typename Tensor>
    int forward_stage1(const Tensor& input,
                       Tensor& mean,
                       Tensor& var,
                       bool is_training)
    {
        set_num_samples(input.get_local_shape()[-1]);
        if (is_training)
        {
            channel_sums_and_sqsums(input, mean, var);
        }
        return 0;
    }

    template <typename Tensor>
    int forward_allreduce(Tensor& mean, Tensor& var, bool is_training)
    {
        if (!is_training || !m_global_stats)
            return 0;

        auto mean_ptr = mean.get_buffer();
        auto var_ptr = var.get_buffer();
        auto count = mean.get_local_pitched_size();
        assert_eq(count, var.get_local_pitched_size());

        // Combine allreduces of mean and var if possible
        if (mean_ptr + count == var_ptr)
        {
            // var comes immediately after mean
            m_allreducer->allreduce(mean_ptr, count * 2);
        }
        else if (mean_ptr == var_ptr + count)
        {
            // mean comes immediately after var
            m_allreducer->allreduce(var_ptr, count * 2);
        }
        else
        {
            m_allreducer->allreduce(mean_ptr, count);
            m_allreducer->allreduce(var_ptr, count);
        }

        return 0;
    }

    template <typename Tensor>
    int forward_stage2(const Tensor& input,
                       Tensor& mean,
                       Tensor& var,
                       Tensor& running_mean,
                       Tensor& running_var,
                       Tensor& scale,
                       Tensor& bias,
                       Tensor& output,
                       bool is_training)
    {
        if (is_training)
        {
            // the sample dimension of the input tensor is assumed to be
            // properly reshaped if necessary (e.g., for the last mini batch
            // in an epoch)
            auto stat_shape =
                m_global_stats ? input.get_shape() : input.get_local_shape();
            // Number of elements per channel. Note that the channel
            // dimension is assumed to be at the second to last dimension.
            index_t num_per_sum = stat_shape.get_size() / stat_shape[-2];

            // Sums to statistics
            sums_to_statistics(
                num_per_sum, mean, var, running_mean, running_var);
            batch_normalization(input, mean, var, scale, bias, output);
        }
        else
        {
            batch_normalization(
                input, running_mean, running_var, scale, bias, output);
        }

        return 0;
    }

#ifdef DISTCONV_HAS_NVSHMEM
    template <typename Tensor>
    int forward_all(const Tensor& input,
                    Tensor& mean,
                    Tensor& var,
                    Tensor& running_mean,
                    Tensor& running_var,
                    Tensor& scale,
                    Tensor& bias,
                    Tensor& output,
                    bool is_training)
    {
        set_num_samples(input.get_local_shape()[-1]);
        if (is_training)
        {
            batchnorm::forward_all<Tensor>(
                m_num_dims,
                input,
                mean,
                var,
                running_mean,
                running_var,
                scale,
                bias,
                output,
                m_decay,
                m_epsilon,
                m_stream,
                *static_cast<tensor::AllreduceNVSHMEM<DataType>*>(
                    m_allreducer.get()));
        }
        else
        {
            util::MPIRootPrintStreamError() << "Not supported";
            return -1;
        }

        return 0;
    }
#endif

    template <typename Tensor>
    int forward(const Tensor& input,
                Tensor& mean,
                Tensor& var,
                Tensor& running_mean,
                Tensor& running_var,
                Tensor& scale,
                Tensor& bias,
                Tensor& output,
                bool is_training)
    {
        util::MPIPrintStreamDebug()
            << "BatchNormalization: " << input << ", " << output;
#ifdef DISTCONV_HAS_NVSHMEM
        if (m_impl == BatchnormImpl::FUSED_NVSHMEM_RECURSIVE_DOUBLING)
        {
            forward_all(input,
                        mean,
                        var,
                        running_mean,
                        running_var,
                        scale,
                        bias,
                        output,
                        is_training);
            return 0;
        }
#endif // DISTCONV_HAS_NVSHMEM
        forward_stage1(input, mean, var, is_training);
        forward_allreduce(mean, var, is_training);
        forward_stage2(input,
                       mean,
                       var,
                       running_mean,
                       running_var,
                       scale,
                       bias,
                       output,
                       is_training);
        return 0;
    }

    template <typename Tensor>
    int backward_stage1(const Tensor& input,
                        const Tensor& d_output,
                        const Tensor& mean,
                        const Tensor& var,
                        const Tensor& scale,
                        Tensor& scale_gradient,
                        Tensor& bias_gradient,
                        Tensor& mean_gradient,
                        Tensor& var_gradient)
    {
        util::MPIPrintStreamDebug() << "BatchNormalization BP stage 1";
        set_num_samples(input.get_local_shape()[-1]);
        backprop1(input,
                  d_output,
                  mean,
                  var,
                  scale,
                  scale_gradient,
                  bias_gradient,
                  mean_gradient,
                  var_gradient);
        return 0;
    }

    template <typename Tensor>
    int backward_allreduce(Tensor& scale_gradient,
                           Tensor& bias_gradient,
                           Tensor& mean_gradient,
                           Tensor& var_gradient,
                           bool skip_weights = false)
    {
        if (!m_global_stats)
            return 0;

        auto mean_ptr = mean_gradient.get_buffer();
        auto var_ptr = var_gradient.get_buffer();
        auto count = mean_gradient.get_local_pitched_size();
        assert_eq(count, var_gradient.get_local_pitched_size());

        if (mean_ptr + count == var_ptr)
        {
            // var comes immediately after mean
            m_allreducer->allreduce(mean_ptr, count * 2);
        }
        else if (mean_ptr == var_ptr + count)
        {
            // mean comes immediately after var
            m_allreducer->allreduce(var_ptr, count * 2);
        }
        else
        {
            m_allreducer->allreduce(mean_ptr, count);
            m_allreducer->allreduce(var_ptr, count);
        }

        if (!skip_weights)
        {
            m_allreducer->allreduce(scale_gradient.get_buffer(),
                                    scale_gradient.get_local_pitched_size());
            m_allreducer->allreduce(bias_gradient.get_buffer(),
                                    bias_gradient.get_local_pitched_size());
        }

        return 0;
    }

    template <typename Tensor>
    int backward_stage2(const Tensor& input,
                        const Tensor& d_output,
                        const Tensor& mean,
                        const Tensor& var,
                        const Tensor& scale,
                        const Tensor& mean_gradient,
                        const Tensor& var_gradient,
                        Tensor& d_input)
    {
        util::MPIPrintStreamDebug() << "BatchNormalization BP stage 2";

        auto stat_shape =
            m_global_stats ? input.get_shape() : input.get_local_shape();
        // Number of elements per channel. Note that the channel
        // dimension is assumed to be at the second to last dimension.
        index_t num_per_sum = stat_shape.get_size() / stat_shape[-2];

        backprop2(num_per_sum,
                  input,
                  d_output,
                  mean,
                  var,
                  scale,
                  mean_gradient,
                  var_gradient,
                  d_input);
        return 0;
    }

    template <typename Tensor>
    int backward(const Tensor& input,
                 const Tensor& d_output,
                 const Tensor& mean,
                 const Tensor& var,
                 const Tensor& scale,
                 Tensor& scale_gradient,
                 Tensor& bias_gradient,
                 Tensor& mean_gradient,
                 Tensor& var_gradient,
                 Tensor& d_input)
    {
        backward_stage1(input,
                        d_output,
                        mean,
                        var,
                        scale,
                        scale_gradient,
                        bias_gradient,
                        mean_gradient,
                        var_gradient);
        backward_allreduce(
            scale_gradient, bias_gradient, mean_gradient, var_gradient);
        backward_stage2(input,
                        d_output,
                        mean,
                        var,
                        scale,
                        mean_gradient,
                        var_gradient,
                        d_input);
        return 0;
    }

    // n: the number of the current local minibatch samples
    void set_num_samples(int n)
    {
        if (n != m_num_current_samples)
        {
            util::MPIPrintStreamDebug()
                << "Changing number of samples to " << n;
        }
        m_num_current_samples = n;
    }

private:
    h2::gpu::DeviceStream m_stream;
    int m_num_dims;
    int m_num_current_samples;
    DataType m_decay;
    DataType m_epsilon;
    std::unique_ptr<tensor::Allreduce<DataType>> m_allreducer;
    BatchnormImpl m_impl;
    bool m_global_stats;

    template <typename Tensor>
    void channel_sums_and_sqsums(const Tensor& input, Tensor& mean, Tensor& var)
    {
        batchnorm::channel_sums_and_sqsums<Tensor>(
            m_num_dims, m_num_current_samples, input, mean, var, m_stream);
    }

    template <typename Tensor>
    void sums_to_statistics(index_t num_per_sum,
                            Tensor& mean,
                            Tensor& var,
                            Tensor& running_mean,
                            Tensor& running_var)
    {
        batchnorm::sums_to_statistics<Tensor>(num_per_sum,
                                              m_decay,
                                              mean,
                                              var,
                                              running_mean,
                                              running_var,
                                              m_stream);
    }

    template <typename Tensor>
    void batch_normalization(const Tensor& input,
                             const Tensor& mean,
                             const Tensor& var,
                             const Tensor& scale,
                             const Tensor& bias,
                             Tensor& output)
    {
        batchnorm::batch_normalization<Tensor>(m_num_dims,
                                               m_num_current_samples,
                                               input,
                                               mean,
                                               var,
                                               scale,
                                               bias,
                                               output,
                                               m_epsilon,
                                               m_stream);
    }

    template <typename Tensor>
    void backprop1(const Tensor& input,
                   const Tensor& d_output,
                   const Tensor& mean,
                   const Tensor& var,
                   const Tensor& scale,
                   Tensor& scale_gradient,
                   Tensor& bias_gradient,
                   Tensor& mean_gradient,
                   Tensor& var_gradient)
    {
        batchnorm::backprop1<Tensor>(m_num_dims,
                                     m_num_current_samples,
                                     input,
                                     d_output,
                                     mean,
                                     var,
                                     scale,
                                     scale_gradient,
                                     bias_gradient,
                                     mean_gradient,
                                     var_gradient,
                                     m_epsilon,
                                     m_stream);
    }

    template <typename Tensor>
    void backprop2(index_t num_per_sum,
                   const Tensor& input,
                   const Tensor& d_output,
                   const Tensor& mean,
                   const Tensor& var,
                   const Tensor& scale,
                   const Tensor& mean_gradient,
                   const Tensor& var_gradient,
                   Tensor& d_input)
    {
        batchnorm::backprop2<Tensor>(m_num_dims,
                                     m_num_current_samples,
                                     num_per_sum,
                                     input,
                                     d_output,
                                     mean,
                                     var,
                                     scale,
                                     mean_gradient,
                                     var_gradient,
                                     d_input,
                                     m_epsilon,
                                     m_stream);
    }
};

} // namespace distconv
