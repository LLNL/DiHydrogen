#pragma once

#include "distconv/base.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/memory_gpu.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_gpu.hpp"

#include <Al.hpp>

namespace distconv
{
namespace tensor
{

template <typename DataType>
class ChannelExchange
{
public:
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;

  ChannelExchange() {}

  ChannelExchange(ChannelExchange<DataType> const& x) {}

  ChannelExchange& operator=(ChannelExchange const& x) { return *this; }

  virtual ~ChannelExchange() {}

  /**
   * Reduce-scatter src by channels into dst.
   *
   * This does the equivalent of running a reduce-scatter on each sample in
   * src.
   */
  virtual void reduce_scatter(TensorType& src,
                              TensorType& dst,
                              Al::NCCLBackend::comm_type& comm,
                              h2::gpu::DeviceStream stream)
  {
    // If there is only one sample, we can do this directly.
    if (src.get_local_shape()[-1] == 1)
    {
      Al::Reduce_scatter<Al::NCCLBackend, DataType>(src.get_base_ptr(),
                                                    dst.get_base_ptr(),
                                                    get_sample_size(dst),
                                                    Al::ReductionOperator::sum,
                                                    comm);
      return;
    }

    DataType* src_buf =
      (DataType*) distconv::internal::RuntimeGPU::get_device_memory_pool().get(
        src.get_local_size() * sizeof(DataType), stream);
    // Pack src such that we can reduce-scatter directly into dst.
    pack_for_rs(src, dst, src_buf, comm.size(), stream);
    Al::Reduce_scatter<Al::NCCLBackend, DataType>(src_buf,
                                                  dst.get_base_ptr(),
                                                  dst.get_local_size(),
                                                  Al::ReductionOperator::sum,
                                                  comm);
    distconv::internal::RuntimeGPU::get_device_memory_pool().release(src_buf);
  }

  virtual void allgather(TensorType& src,
                         TensorType& dst,
                         Al::NCCLBackend::comm_type& comm,
                         h2::gpu::DeviceStream stream)
  {
    // If there is only one sample, we can do this directly.
    if (src.get_local_shape()[-1] == 1)
    {
      Al::Allgather<Al::NCCLBackend, DataType>(
        src.get_base_ptr(), dst.get_base_ptr(), get_sample_size(src), comm);
      return;
    }

    DataType* dst_buf =
      (DataType*) distconv::internal::RuntimeGPU::get_device_memory_pool().get(
        dst.get_local_size() * sizeof(DataType), stream);
    Al::Allgather<Al::NCCLBackend, DataType>(
      src.get_base_ptr(), dst_buf, src.get_local_size(), comm);
    // Unpack dst, which is interleaved.
    unpack_from_ag(src, dst, dst_buf, comm.size(), stream);
    distconv::internal::RuntimeGPU::get_device_memory_pool().release(dst_buf);
  }

protected:
  index_t get_sample_size(TensorType const& t) const
  {
    IndexVector idx = IndexVector(t.get_num_dims(), 0);
    idx[-1] = 1;
    return t.get_local_offset(idx);
  }

  index_t get_channel_size(TensorType const& t) const
  {
    IndexVector idx = IndexVector(t.get_num_dims(), 0);
    idx[-2] = 1;
    return t.get_local_offset(idx);
  }

  void pack_for_rs(TensorType& src,
                   TensorType& dst,
                   DataType* dst_buf,
                   size_t comm_size,
                   h2::gpu::DeviceStream stream);

  void unpack_from_ag(TensorType& src,
                      TensorType& dst,
                      DataType* packed_buf,
                      size_t comm_size,
                      h2::gpu::DeviceStream stream);
};

}  // namespace tensor
}  // namespace distconv
