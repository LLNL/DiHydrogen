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
        auto const& handle = m_be.get_handle();
        // Note: These proxies do not need to be "forced" since cuDNN
        //       only cares about stride information when an in-place
        //       operation is performed, which is never the case here.
        auto input_proxy =
            dnn_lib::read_proxy(handle, m_input_d, input.get_const_base_ptr());
        auto output_proxy = dnn_lib::write_proxy(
            handle, m_output_d, output.get_base_ptr(), beta);
        backend::activation_forward(m_be.get_handle(),
                                    m_activation_d,
                                    alpha,
                                    input_proxy.desc(),
                                    input_proxy.ptr(),
                                    beta,
                                    output_proxy.desc(),
                                    output_proxy.ptr());
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
        // The _actual_ requirement here is that "strides(m_output_d)
        // == strides(m_d_output_d) && strids(m_input_d) ==
        // strides(m_d_input_d)" We can't handle that directly, so
        // instead we proxy everything, leaving plenty of room for
        // future optimization.
        auto const& handle = m_be.get_handle();
        auto y_proxy = dnn_lib::force_read_proxy(
            handle, m_output_d, output.get_const_base_ptr());
        auto dy_proxy = dnn_lib::force_read_proxy(
            handle, m_d_output_d, d_output.get_const_base_ptr());
        auto x_proxy = dnn_lib::force_read_proxy(
            handle, m_input_d, input.get_const_base_ptr());
        auto dx_proxy = dnn_lib::force_write_proxy(
            handle, m_d_input_d, d_input.get_base_ptr(), beta);
        backend::activation_backward(handle,
                                     m_activation_d,
                                     alpha,
                                     y_proxy.desc(),
                                     y_proxy.ptr(),
                                     dy_proxy.desc(),
                                     dy_proxy.ptr(),
                                     x_proxy.desc(),
                                     x_proxy.ptr(),
                                     beta,
                                     dx_proxy.desc(),
                                     dx_proxy.ptr());
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
