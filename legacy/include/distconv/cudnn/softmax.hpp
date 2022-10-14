#pragma once

#include "distconv/cudnn/backend.hpp"
#include "distconv/runtime_gpu.hpp"

#include <Al.hpp>

#include <memory>

namespace distconv
{

template <>
class Softmax<BackendDNNLib>
{
public:
    Softmax(BackendDNNLib& backend) : m_be(backend) {}

    ~Softmax() = default;

    Softmax& operator=(const Softmax& x)
    {
        assert_always(&m_be == &x.m_be);
        return *this;
    }

    template <typename Tensor>
    void setup(const Tensor& input, SoftmaxMode mode)
    {
        m_mode = mode;
        auto loc_shape = input.get_locale_shape();
        m_num_procs_per_sample = loc_shape.reduce_sum() / loc_shape[-1];
        if (m_num_procs_per_sample > 1)
        {
            auto sample_loc = input.get_sub_locale_except_dim(-1);
            m_sample_al.reset(new Al::NCCLBackend::comm_type(
                sample_loc.get_comm(), m_be.get_stream()));
        }
    }

    template <typename Tensor>
    int forward(const Tensor& x, Tensor& y);

    template <typename Tensor>
    int backward(const Tensor& y, const Tensor& dy, Tensor& dx);

protected:
    BackendDNNLib& m_be;
    SoftmaxMode m_mode;
    int m_num_procs_per_sample;
    std::unique_ptr<Al::NCCLBackend::comm_type> m_sample_al;

    template <typename DataType>
    void allreduce(DataType* sample_values, int num_samples, bool max_or_sum)
    {
        if (m_num_procs_per_sample < 2)
            return;

        auto op = max_or_sum ? Al::ReductionOperator::max
                             : Al::ReductionOperator::sum;
        Al::Allreduce<Al::NCCLBackend, DataType>(
            sample_values, num_samples, op, *m_sample_al.get());
    }
};

} // namespace distconv
