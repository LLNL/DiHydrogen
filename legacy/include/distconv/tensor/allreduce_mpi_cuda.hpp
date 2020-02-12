#pragma once

#include "distconv/tensor/allreduce.hpp"
#include "distconv/tensor/allreduce_mpi.hpp"
#include "distconv/tensor/runtime_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"

namespace distconv {
namespace tensor {

template <typename DataType>
class AllreduceMPICUDA: public AllreduceMPI<DataType> {
 public:
  AllreduceMPICUDA(MPI_Comm comm, cudaStream_t stream):
      AllreduceMPI<DataType>(comm), m_stream(stream) {}
  virtual ~AllreduceMPICUDA() = default;

  using AllreduceMPI<DataType>::allreduce;

  virtual void allreduce(const DataType *send_buf, DataType *recv_buf,
                         size_t count) override {
    assert_always(send_buf != nullptr);
    assert_always(recv_buf != nullptr);
    assert_always(count > 0);
    auto &x = internal::RuntimeCUDA::get_pinned_memory_pool();
    auto len = sizeof(DataType) * count;
    DataType *host_buf = static_cast<DataType*>(x.get(len));
    assert_always(host_buf != nullptr);
    DISTCONV_CHECK_CUDA(cudaMemcpyAsync(host_buf, send_buf, len,
                                        cudaMemcpyDeviceToHost,
                                        m_stream));
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(m_stream));
    AllreduceMPI<DataType>::allreduce(host_buf, host_buf, count);
    DISTCONV_CHECK_CUDA(cudaMemcpyAsync(recv_buf, host_buf, len,
                                        cudaMemcpyHostToDevice,
                                        m_stream));
    // Sync the stream before releasing the pinned buffer
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(m_stream));
    x.release(host_buf);
  }

 protected:
  cudaStream_t m_stream;
};

} // namespace tensor
} // namespace distconv
