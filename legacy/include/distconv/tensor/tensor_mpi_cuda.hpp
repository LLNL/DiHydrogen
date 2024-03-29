#pragma once

#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/runtime_gpu.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_gpu.hpp"
#include "distconv/util/util_mpi.hpp"
#include <distconv_config.hpp>

#if H2_HAS_CUDA
#define GPU_MEMCPY_3D_PARAMS cudaMemcpy3DParms
#elif H2_HAS_ROCM
#define GPU_MEMCPY_3D_PARAMS hipMemcpy3DParms
#endif

namespace distconv {
namespace tensor {
namespace internal {

template <typename DataType>
struct HostShadow<Tensor<DataType, LocaleMPI, CUDAAllocator>> {
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  using ShadowTensorType = Tensor<DataType, LocaleMPI,
                                  CUDAHostPooledAllocator>;

  HostShadow() = delete;

  HostShadow(const TensorType &tensor):
      m_tensor(tensor),
      m_shadow(tensor.get_shape(), tensor.get_locale(),
               tensor.get_distribution().get_non_overlapped_distribution(),
               tensor.get_requested_local_shape(),
               tensor.get_requested_local_block()) {
    assert0(m_shadow.allocate());
  }

  const ShadowTensorType &get_host_shadow() const {
    return m_shadow;
  }

  ShadowTensorType &get_host_shadow() {
    return m_shadow;
  }

  void sync_from_dev() {
    util::MPIPrintStreamDebug() << "Copy to host";
    Copy(m_shadow, m_tensor);
  }

  void sync_to_dev() {
    util::MPIPrintStreamDebug() << "Copy back to device";
    Copy(m_tensor, m_shadow);
  }

 protected:
  TensorType m_tensor;
  ShadowTensorType m_shadow;
};

template <typename DataType>
struct HostShadow<Tensor<DataType, LocaleMPI,
                         CUDAHostPooledAllocator>> {
  using TensorType = Tensor<DataType, LocaleMPI,
                            CUDAHostPooledAllocator>;
  using ShadowTensorType = TensorType;

  HostShadow() = delete;

  HostShadow(const TensorType &tensor): m_tensor(tensor) {}

  const ShadowTensorType &get_host_shadow() const {
    return m_tensor;
  }

  ShadowTensorType &get_host_shadow() {
    return m_tensor;
  }

  void sync_to_dev() {}
  void sync_from_dev() {}

 protected:
  TensorType m_tensor;
};


// Specialization for CUDA objects
template <typename DataType, typename StreamType>
struct CopyLocalFunctor3D<DataType, CUDAAllocator, CUDAAllocator,
                          StreamType> {

  int operator()(Tensor<DataType, LocaleMPI, CUDAAllocator> &t_dst,
                 const Tensor<DataType, LocaleMPI, CUDAAllocator> &t_src,
                 StreamType stream) {
    // Use cudaMemcpy3D
    const int nd = t_src.get_num_dims();
    assert_always(nd >= 3);
    util::MPIPrintStreamDebug()
        << "CopyLocal from " << t_src << " to " << t_dst;
    const auto &local_shape = t_src.get_local_shape();
    assert_eq(local_shape, t_dst.get_local_shape());
    auto tr_shape = local_shape;
    tr_shape[0] = 1;
    tr_shape[1] = 1;
    tr_shape[2] = 1;
    const size_t num_chunks = tr_shape.get_size();
    int src_offset = 0;
    int dst_offset = 0;
    for (int i = 3; i < nd; ++i) {
      src_offset += t_src.get_overlap()[i] * tr_shape[i-1];
      dst_offset += t_dst.get_overlap()[i] * tr_shape[i-1];
    }
    GPU_MEMCPY_3D_PARAMS p;
    memset(&p, 0, sizeof(GPU_MEMCPY_3D_PARAMS));
    p.extent = GPU_MAKE_GPU_EXTENT(
        local_shape[0] * sizeof(DataType), local_shape[1], local_shape[2]);
    // cudaPitchedPtr does not have const void *
    p.srcPtr = GPU_MAKE_GPU_PITCHED_PTR(
        const_cast<void*>(static_cast<const void*>(t_src.get_const_buffer())),
        t_src.get_pitch() * sizeof(DataType),
        t_src.get_local_real_shape()[0],
        t_src.get_local_real_shape()[1]);
    p.dstPtr = GPU_MAKE_GPU_PITCHED_PTR(t_dst.get_buffer(),
                                        t_dst.get_pitch() * sizeof(DataType),
                                        t_dst.get_local_real_shape()[0],
                                        t_dst.get_local_real_shape()[1]);
    p.kind = GPU_MEMCPY_DEFAULT;
    for (size_t i = 0; i < num_chunks; ++i) {
        p.srcPos = GPU_MAKE_GPU_POS(t_src.get_overlap()[0] * sizeof(DataType),
                                    t_src.get_overlap()[1],
                                    t_src.get_overlap()[2]
                                        + t_src.get_local_real_shape()[2]
                                              * (i + src_offset));
        p.dstPos = GPU_MAKE_GPU_POS(t_dst.get_overlap()[0] * sizeof(DataType),
                                    t_dst.get_overlap()[1],
                                    t_dst.get_overlap()[2]
                                        + t_dst.get_local_real_shape()[2]
                                              * (i + dst_offset));
        // util::MPIPrintStreamDebug() << "memcpy3d param: " << p << "\n";
        DISTCONV_CHECK_GPU(GPU_MEMCPY_3D_ASYNC(&p, get_gpu_stream(stream)));
    }
    return 0;
  }
};

} // namespace internal

template <typename DataType>
class TensorImplHelper<DataType, CUDAAllocator> {
  using TensorImplType = TensorImpl<Tensor<DataType, LocaleMPI, CUDAAllocator>>;
 public:
  TensorImplHelper(TensorImplType &impl): m_impl(impl) {}
  void clear_halo(int dim, h2::gpu::DeviceStream s);
  void scale(DataType v, h2::gpu::DeviceStream s);

  protected:
  TensorImplType &m_impl;
};

template <typename DataType1, typename DataType2>
int Cast(Tensor<DataType1, LocaleMPI, CUDAAllocator>& t_dest,
         const Tensor<DataType2, LocaleMPI, CUDAAllocator>& t_src,
         h2::gpu::DeviceStream s);

template <typename DataType>
inline int Cast(Tensor<DataType, LocaleMPI, CUDAAllocator>& t_dest,
                const Tensor<DataType, LocaleMPI, CUDAAllocator>& t_src,
                h2::gpu::DeviceStream s)
{
    return Copy(t_dest, t_src, s);
}

template <typename DataType1, typename DataType2>
int CastScaleBias(Tensor<DataType1, LocaleMPI, CUDAAllocator>& t_dest,
                  const Tensor<DataType2, LocaleMPI, CUDAAllocator>& t_src,
                  const DataType1 alpha,
                  const DataType1 beta,
                  h2::gpu::DeviceStream s);

template <typename DataType>
int Concatenate(Tensor<DataType, LocaleMPI, CUDAAllocator>& t_dest,
                const Tensor<DataType, LocaleMPI, CUDAAllocator>& t_src1,
                const Tensor<DataType, LocaleMPI, CUDAAllocator>& t_src2,
                h2::gpu::DeviceStream s);

template <typename DataType>
int Slice(Tensor<DataType, LocaleMPI, CUDAAllocator>& t_dest1,
          Tensor<DataType, LocaleMPI, CUDAAllocator>& t_dest2,
          const Tensor<DataType, LocaleMPI, CUDAAllocator>& t_src,
          h2::gpu::DeviceStream s);

} // namespace tensor
} // namespace distconv
