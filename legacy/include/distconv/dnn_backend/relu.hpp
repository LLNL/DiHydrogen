#pragma once

#include "distconv/layers.hpp"
#include "distconv/dnn_backend/backend.hpp"
#include "distconv/runtime_gpu.hpp"

namespace distconv
{

template <>
class ReLU<DNNBackend<GPUDNNBackend>>
{
public:
    ReLU(DNNBackend<GPUDNNBackend>& backend)
        : m_be(backend),
          m_activation_d{GPUDNNBackend::make_activation_descriptor()},
          m_input_d{GPUDNNBackend::make_tensor_descriptor()},
          m_output_d{GPUDNNBackend::make_tensor_descriptor()},
          m_d_input_d{GPUDNNBackend::make_tensor_descriptor()},
          m_d_output_d{GPUDNNBackend::make_tensor_descriptor()}
    {}

    ~ReLU()
    {
        GPUDNNBackend::destroy_tensor_descriptor(m_d_output_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_d_input_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_output_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_input_d);
        GPUDNNBackend::destroy_activation_descriptor(m_activation_d);
    }

    template <typename Tensor, typename ConstTensor>
    void setup(const ConstTensor& input,
               const Tensor& output,
               const Tensor& d_input,
               const ConstTensor& d_output)
    {
        GPUDNNBackend::setup_tensor_descriptor(m_input_d, input, false);
        GPUDNNBackend::setup_tensor_descriptor(m_output_d, output, false);
        GPUDNNBackend::setup_tensor_descriptor(m_d_input_d, d_input, false);
        GPUDNNBackend::setup_tensor_descriptor(m_d_output_d, d_output, false);
        GPUDNNBackend::setup_relu_activation_descriptor(m_activation_d);
    }

    template <typename Tensor>
    int forward(typename Tensor::data_type alpha,
                const Tensor& input,
                typename Tensor::data_type beta,
                Tensor& output)
    {
        util::MPIPrintStreamDebug()
            << "Relu FP: " << m_input_d << ", " << m_output_d
            << ", input ptr: " << input.get_const_base_ptr()
            << ", output ptr: " << output.get_base_ptr() << "\n";
        if (input.get_local_size() == 0)
        {
            return 0;
        }
        set_num_samples(input.get_local_shape()[-1]);
        m_be.activation_forward(m_activation_d,
                                alpha,
                                m_input_d,
                                input.get_const_base_ptr(),
                                beta,
                                m_output_d,
                                output.get_base_ptr());
        return 0;
    }

    template <typename Tensor>
    int backward(typename Tensor::data_type alpha,
                 Tensor& output,
                 const Tensor& d_output,
                 const Tensor& input,
                 typename Tensor::data_type beta,
                 Tensor& d_input)
    {
        util::MPIPrintStreamDebug()
            << "Relu BP: " << m_input_d << ", " << m_d_output_d << ", "
            << m_output_d << ", " << m_d_input_d;
        if (d_input.get_local_size() == 0)
        {
            return 0;
        }
        set_num_samples(d_input.get_local_shape()[-1]);
        m_be.activation_backward(m_activation_d,
                                 alpha,
                                 m_output_d,
                                 output.get_const_base_ptr(),
                                 m_d_output_d,
                                 d_output.get_const_base_ptr(),
                                 m_input_d,
                                 d_output.get_const_base_ptr(),
                                 beta,
                                 m_d_input_d,
                                 d_input.get_base_ptr());
        return 0;
    }

    void set_num_samples(int n)
    {
        if (n != GPUDNNBackend::get_tensor_num_samples(m_input_d))
        {
            GPUDNNBackend::set_tensor_num_samples(m_input_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_output_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_d_input_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_d_output_d, n);
        }
    }

private:
    DNNBackend<GPUDNNBackend>& m_be;
    GPUDNNBackend::ActivationDescriptor_t m_activation_d;
    GPUDNNBackend::TensorDescriptor_t m_input_d;
    GPUDNNBackend::TensorDescriptor_t m_output_d;
    GPUDNNBackend::TensorDescriptor_t m_d_input_d;
    GPUDNNBackend::TensorDescriptor_t m_d_output_d;
};

} // namespace distconv
