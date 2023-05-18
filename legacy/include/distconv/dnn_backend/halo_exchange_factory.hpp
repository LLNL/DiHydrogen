#pragma once

#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda_al.hpp"
#include "distconv/tensor/halo_exchange_cuda_mpi.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/halo_exchange_cuda_hybrid.hpp"
#include "distconv/tensor/halo_exchange_cuda_p2p.hpp"
#endif // DISTCONV_HAS_P2P
#include "distconv/tensor/channel_exchange.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/halo_exchange_cuda_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/tensor_mpi.hpp"

namespace distconv
{

template <typename DataT, typename AllocT>
auto make_halo_exchange(
    tensor::Tensor<DataT, tensor::LocaleMPI, AllocT>& tensor,
#ifdef DISTCONV_HAS_P2P
    p2p::P2P& p2p,
#endif
    HaloExchangeMethod method)
{
    using AlBackend = Al::NCCLBackend;
    using Allocator = tensor::CUDAAllocator; // may differ from AllocT.
    using HaloExchange = tensor:: HaloExchange<DataT, Allocator, AlBackend>;
    using HaloExchangeMPI = tensor::HaloExchangeMPI<DataT,
                                                    Allocator,
                                                    AlBackend>;
    using HaloExchangeAL = tensor::HaloExchangeAL<DataT,
                                                  Allocator,
                                                  AlBackend>;
#ifdef DISTCONV_HAS_P2P
    using HaloExchangeP2P = tensor::HaloExchangeP2P<DataT,
                                                    Allocator,
                                                    AlBackend>;
    using HaloExchangeHybrid =
        tensor::HaloExchangeHybrid<DataT,
                                   Allocator,
                                   AlBackend>;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
    using HaloExchangeNVSHMEM =
        tensor::HaloExchangeNVSHMEM<DataT,
                                    Allocator,
                                    AlBackend>;
    using HaloExchangeNVSHMEMDirect =
        tensor::HaloExchangeNVSHMEMDirect<DataT,
                                          Allocator,
                                          AlBackend>;
    using HaloExchangeNVSHMEMFusedNotify =
        tensor::HaloExchangeNVSHMEMFusedNotify<DataT,
                                               Allocator,
                                               AlBackend>;
#ifdef DISTCONV_HAS_CUDA_GRAPH
    using HaloExchangeNVSHMEMGraph =
        tensor::HaloExchangeNVSHMEMGraph<DataT,
                                         Allocator,
                                         AlBackend>;
#endif // DISTCONV_HAS_CUDA_GRAPH
#endif // DISTCONV_HAS_NVSHMEM

    util::MPIRootPrintStreamDebug()
        << "Using " << method << " in halo exchange";
    std::unique_ptr<HaloExchange> out;
    switch (method)
    {
    case HaloExchangeMethod::MPI:
        out = std::make_unique<HaloExchangeMPI>(tensor);
        break;
    case HaloExchangeMethod::AL:
        out = std::make_unique<HaloExchangeAL>(tensor);
        break;
#ifdef DISTCONV_HAS_P2P
    case HaloExchangeMethod::P2P:
        out = std::make_unique<HaloExchangeP2P>(tensor, p2p);
        break;
    case HaloExchangeMethod::HYBRID:
        out = std::make_unique<HaloExchangeHybrid>(tensor, p2p);
        break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
    case HaloExchangeMethod::NVSHMEM:
        out = std::make_unique<HaloExchangeNVSHMEM>(tensor);
        break;
    case HaloExchangeMethod::NVSHMEM_DIRECT:
        out = std::make_unique<HaloExchangeNVSHMEMDirect>(tensor);
        break;
    case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
        out = std::make_unique<HaloExchangeNVSHMEMFusedNotify>(tensor);
        break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
    case HaloExchangeMethod::NVSHMEM_GRAPH:
        out = std::make_unique<HaloExchangeNVSHMEMGraph>(tensor);
        break;
#endif // DISTCONV_HAS_CUDA_GRAPH
#endif // DISTCONV_HAS_NVSHMEM
    default:
        util::MPIPrintStreamError()
            << "Invalid halo exchange method: " << method;
        std::abort();
    }
    return out;
}

} // namespace distconv
