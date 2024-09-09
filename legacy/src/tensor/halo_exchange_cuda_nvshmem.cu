#include "distconv/tensor/halo_exchange_cuda_nvshmem.hpp"
#include "distconv/tensor/halo_packing_cuda.hpp"

// Definitions only for float and double as integer types are unlikely
// to used.
#define LIST_OF_TYPES                                                          \
  DEFINE_FUNC(float)                                                           \
  DEFINE_FUNC(double)

namespace distconv
{
namespace tensor
{

#define DEFINE_FUNC(TYPE)                                                      \
  template <>                                                                  \
  void HaloExchangeNVSHMEMDirect<TYPE, CUDAAllocator, Al::NCCLBackend>::       \
    pack_and_put(int dim,                                                      \
                 Side side,                                                    \
                 int width,                                                    \
                 cudaStream_t stream,                                          \
                 void* buf,                                                    \
                 bool is_reverse,                                              \
                 void* dst,                                                    \
                 int peer)                                                     \
  {                                                                            \
    halo_exchange_cuda::pack_and_put_block<TYPE>(                              \
      m_tensor, dim, side, width, stream, buf, is_reverse, dst, peer);         \
  }

LIST_OF_TYPES
#undef DEFINE_FUNC

#define DEFINE_FUNC(TYPE)                                                      \
  template <>                                                                  \
  void HaloExchangeNVSHMEMFusedNotify<TYPE, CUDAAllocator, Al::NCCLBackend>::  \
    pack_put_notify(int dim,                                                   \
                    Side side,                                                 \
                    int width,                                                 \
                    cudaStream_t stream,                                       \
                    void* buf,                                                 \
                    bool is_reverse,                                           \
                    void* dst,                                                 \
                    int peer)                                                  \
  {                                                                            \
    auto sync = this->m_sync(dim)[side];                                       \
    halo_exchange_cuda::pack_put_notify_block<TYPE>(                           \
      m_tensor, dim, side, width, stream, buf, is_reverse, dst, peer, sync);   \
  }

LIST_OF_TYPES
#undef DEFINE_FUNC

#define DEFINE_FUNC(TYPE)                                                      \
  template <>                                                                  \
  void HaloExchangeNVSHMEMFusedNotify<TYPE, CUDAAllocator, Al::NCCLBackend>::  \
    wait_and_unpack(int dim,                                                   \
                    Side side,                                                 \
                    int width,                                                 \
                    cudaStream_t stream,                                       \
                    void* buf,                                                 \
                    bool is_reverse,                                           \
                    HaloExchangeAccumOp op)                                    \
  {                                                                            \
    auto sync = this->m_sync(dim)[side];                                       \
    halo_exchange_cuda::wait_and_unpack<TYPE>(                                 \
      m_tensor, dim, side, width, stream, buf, is_reverse, op, sync);          \
  }

LIST_OF_TYPES
#undef DEFINE_FUNC

}  // namespace tensor
}  // namespace distconv
