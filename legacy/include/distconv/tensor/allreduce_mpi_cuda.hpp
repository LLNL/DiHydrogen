#pragma once

#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_mpi.hpp"
#include "distconv/tensor/runtime_gpu.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"

namespace distconv {
namespace tensor {

template <typename DataType>
class AllreduceMPICUDA: public AllreduceMPI<DataType> {
public:
    AllreduceMPICUDA(MPI_Comm comm, h2::gpu::DeviceStream stream)
        : AllreduceMPI<DataType>(comm), m_stream(stream)
    {}
    virtual ~AllreduceMPICUDA() = default;

    using AllreduceMPI<DataType>::allreduce;

    virtual void allreduce(const DataType* send_buf,
                           DataType* recv_buf,
                           size_t count) override
    {
        assert_always(send_buf != nullptr);
        assert_always(recv_buf != nullptr);
        assert_always(count > 0);
        auto& x = internal::RuntimeGPU::get_pinned_memory_pool();
        auto const len = count * sizeof(DataType);
        DataType* host_buf = static_cast<DataType*>(x.get(len));
        assert_always(host_buf != nullptr);
        h2::gpu::mem_copy(host_buf, send_buf, count, m_stream);
        h2::gpu::sync(m_stream);
        AllreduceMPI<DataType>::allreduce(host_buf, host_buf, count);
        h2::gpu::mem_copy(recv_buf, host_buf, count, m_stream);
        // Sync the stream before releasing the pinned buffer
        h2::gpu::sync(m_stream);
        x.release(host_buf);
  }

  private:
  h2::gpu::DeviceStream m_stream;
};

} // namespace tensor
} // namespace distconv
