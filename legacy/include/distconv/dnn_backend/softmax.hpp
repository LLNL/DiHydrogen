#pragma once

#include "distconv/dnn_backend/backend.hpp"
#include "distconv/layers.hpp"
#include "distconv/runtime_gpu.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv
{

template <>
class Softmax<DNNBackend<GPUDNNBackend>>
{
public:
    Softmax(DNNBackend<GPUDNNBackend> const& backend)
        : m_stream(backend.get_stream())
    {}

    Softmax(h2::gpu::DeviceStream stream) : m_stream{stream} {}

    ~Softmax() = default;

    template <typename Tensor>
    void setup(const Tensor& input, SoftmaxMode mode)
    {
        m_mode = mode;
        auto loc_shape = input.get_locale_shape();
        m_num_procs_per_sample = loc_shape.reduce_sum() / loc_shape[-1];
        if (m_num_procs_per_sample > 1)
        {
            auto sample_loc = input.get_sub_locale_except_dim(-1);
            m_sample_al = std::make_unique<Al::NCCLBackend::comm_type>(
                sample_loc.get_comm(), m_stream);
        }
    }

    template <typename Tensor>
    int forward(const Tensor& x, Tensor& y);

    template <typename Tensor>
    int backward(const Tensor& y, const Tensor& dy, Tensor& dx);

private:
    h2::gpu::DeviceStream m_stream;
    SoftmaxMode m_mode;
    int m_num_procs_per_sample;
    std::unique_ptr<Al::NCCLBackend::comm_type> m_sample_al;

    template <typename DataType>
    void allreduce(DataType* sample_values, int num_samples, bool max_or_sum)
    {
        if (m_num_procs_per_sample < 2)
            return;

        auto const op = max_or_sum ? Al::ReductionOperator::max
                                   : Al::ReductionOperator::sum;
        Al::Allreduce<Al::NCCLBackend, DataType>(
            sample_values, num_samples, op, *m_sample_al.get());
    }
};

} // namespace distconv
