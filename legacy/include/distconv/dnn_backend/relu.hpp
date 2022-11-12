#pragma once

#include "distconv/dnn_backend/backend.hpp"
#include "distconv/runtime_gpu.hpp"

namespace distconv
{

template <>
class ReLU<BackendDNNLib>
{
public:
    ReLU(BackendDNNLib& backend)
        : m_be(backend),
          m_activation_d{backend::make_activation_descriptor()},
          m_input_d{backend::make_tensor_descriptor()},
          m_output_d{backend::make_tensor_descriptor()},
          m_d_input_d{backend::make_tensor_descriptor()},
          m_d_output_d{backend::make_tensor_descriptor()}
    {}

    ~ReLU()
    {
        backend::destroy_tensor_descriptor(m_d_output_d);
        backend::destroy_tensor_descriptor(m_d_input_d);
        backend::destroy_tensor_descriptor(m_output_d);
        backend::destroy_tensor_descriptor(m_input_d);
        backend::destroy_activation_descriptor(m_activation_d);
    }

    ReLU<BackendDNNLib> operator=(const ReLU<BackendDNNLib>& x)
    {
        assert_always(&m_be == &x.m_be);
        backend::copy_tensor_descriptor(m_input_d, x.m_input_d);
        backend::copy_tensor_descriptor(m_output_d, x.m_output_d);
        backend::copy_tensor_descriptor(m_d_input_d, x.m_d_input_d);
        backend::copy_tensor_descriptor(m_d_output_d, x.m_d_output_d);
        backend::copy_activation_descriptor(m_activation_d, x.m_activation_d);
        return *this;
    }

    template <typename Tensor, typename ConstTensor>
    void setup(const ConstTensor& input,
               const Tensor& output,
               const Tensor& d_input,
               const ConstTensor& d_output)
    {
        backend::setup_tensor_descriptor(m_input_d, input, false);
        backend::setup_tensor_descriptor(m_output_d, output, false);
        backend::setup_tensor_descriptor(m_d_input_d, d_input, false);
        backend::setup_tensor_descriptor(m_d_output_d, d_output, false);
        backend::setup_relu_activation_descriptor(m_activation_d);
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
        backend::activation_forward(m_be.get_handle(),
                                    m_activation_d,
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
        backend::activation_backward(m_be.get_handle(),
                                     m_activation_d,
                                     alpha,
                                     m_output_d,
                                     output.get_const_base_ptr(),
                                     m_d_output_d,
                                     d_output.get_const_base_ptr(),
                                     m_input_d,
                                     input.get_const_base_ptr(),
                                     beta,
                                     m_d_input_d,
                                     d_input.get_base_ptr());
        return 0;
    }

    void set_num_samples(int n)
    {
        if (n != backend::get_tensor_num_samples(m_input_d))
        {
            backend::set_tensor_num_samples(m_input_d, n);
            backend::set_tensor_num_samples(m_output_d, n);
            backend::set_tensor_num_samples(m_d_input_d, n);
            backend::set_tensor_num_samples(m_d_output_d, n);
        }
    }

protected:
    BackendDNNLib& m_be;
    backend::ActivationDescriptor_t m_activation_d;
    backend::TensorDescriptor_t m_input_d;
    backend::TensorDescriptor_t m_output_d;
    backend::TensorDescriptor_t m_d_input_d;
    backend::TensorDescriptor_t m_d_output_d;
};

} // namespace distconv
