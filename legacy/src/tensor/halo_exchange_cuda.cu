#include "distconv/tensor/halo_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_packing_cuda.hpp"
#include "distconv/util/util_mpi.hpp"

#include <limits>

namespace distconv
{
namespace tensor
{

template <>
void HaloExchange<float, CUDAAllocator, Al::NCCLBackend>::pack_or_unpack(
  int dim,
  Side side,
  int width,
  h2::gpu::DeviceStream stream,
  void* buf,
  bool is_pack,
  bool is_reverse,
  HaloExchangeAccumOp op)
{
  halo_exchange_cuda::pack_or_unpack<float>(
    m_tensor, dim, side, width, stream, buf, is_pack, is_reverse, op);
}

template <>
void HaloExchange<double, CUDAAllocator, Al::NCCLBackend>::pack_or_unpack(
  int dim,
  Side side,
  int width,
  h2::gpu::DeviceStream stream,
  void* buf,
  bool is_pack,
  bool is_reverse,
  HaloExchangeAccumOp op)
{
  halo_exchange_cuda::pack_or_unpack<double>(
    m_tensor, dim, side, width, stream, buf, is_pack, is_reverse, op);
}

}  // namespace tensor
}  // namespace distconv
