#pragma once

#include "distconv/distconv.hpp"
#include "distconv/layers.hpp"
#include "distconv/dnn_backend/dnn_backend.hpp"
#include "distconv/dnn_backend/pack_unpack.hpp"
#include "distconv/runtime_gpu.hpp"
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda_al.hpp"
#include "distconv/tensor/halo_exchange_cuda_mpi.hpp"
#include "distconv/util/util.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/halo_exchange_cuda_hybrid.hpp"
#include "distconv/tensor/halo_exchange_cuda_p2p.hpp"
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/halo_exchange_cuda_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM

namespace distconv
{

namespace pooling
{
#if H2_HAS_CUDA
inline cudnnPoolingMode_t get_pooling_mode(const std::string& name,
                                           bool deterministic)
{
    if (name == "MAX")
    {
        // This does not seem to be necessary. It's not clear what the
        // difference of the two algorithms is.
        if (deterministic)
            return CUDNN_POOLING_MAX_DETERMINISTIC;
        else
            return CUDNN_POOLING_MAX;
    }
    else if (name == "AVERAGE")
        return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    else if (name == "AVERAGE_NO_PAD")
        return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    else
    {
        util::PrintStreamError()
            << "No matching pooling mode found for CUDNN: " << name << "\n";
        std::abort();
    }
}
#elif H2_HAS_ROCM
inline miopenPoolingMode_t get_pooling_mode(std::string const& name,
                                            bool /*deterministic*/)
{
    if (name == "MAX")
        return miopenPoolingMax;
    else if (name == "AVERAGE")
        return miopenPoolingAverageInclusive;
    else if (name == "AVERAGE_NO_PAD")
        return miopenPoolingAverage;
    else
    {
        util::PrintStreamError()
            << "No matching pooling mode found for MIOpen: " << name << "\n";
        std::abort();
    }
    return miopenPoolingMax;
}
#endif
} // namespace pooling

template <typename DataType>
class Pooling<DNNBackend<GPUDNNBackend>, DataType>
{
    using LocaleMPI = tensor::LocaleMPI;

public:
    Pooling(DNNBackend<GPUDNNBackend>& backend,
            int num_dims,
            HaloExchangeMethod method)
        : m_be(backend),
          m_num_dims(num_dims),
          m_num_spatial_dims(num_dims - 2),
          m_input_d{GPUDNNBackend::make_tensor_descriptor()},
          m_output_d{GPUDNNBackend::make_tensor_descriptor()},
          m_d_input_d{GPUDNNBackend::make_tensor_descriptor()},
          m_d_output_d{GPUDNNBackend::make_tensor_descriptor()},
          m_pooling_d{GPUDNNBackend::make_pooling_descriptor()},
          m_halo_xch_method(method)
    {}

    ~Pooling()
    {
        GPUDNNBackend::destroy_pooling_descriptor(m_pooling_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_d_output_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_d_input_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_output_d);
        GPUDNNBackend::destroy_tensor_descriptor(m_input_d);
    }

    template <typename Tensor>
    void setup(Tensor& input,
               Tensor& output,
               Tensor& d_input,
               Tensor& d_output,
               int_vector windows,
               int_vector pads,
               int_vector strides,
               const std::string& mode)
    {
        {
            // All of the dimensions must be the same
            assert_eq((unsigned int) m_num_spatial_dims, windows.size());
            assert_eq((unsigned int) m_num_spatial_dims, pads.size());
            assert_eq((unsigned int) m_num_spatial_dims, strides.size());
        }

        // TODO: asymmetric not supported
        assert_always(util::is_all_elements_equal(windows));
        assert_always(util::is_all_elements_equal(pads));
        assert_always(util::is_all_elements_equal(strides));

        // TODO: only stencil-like windows are supported
        for (int i = 0; i < m_num_spatial_dims; ++i)
        {
            auto w = windows[i];
            if (w % 2)
            {
                auto stencil = (w - 1) / 2;
                // Padding must be zero or match with the stencil size
                assert_always(pads[i] == 0 || pads[i] == stencil);
                // TODO: stride limitation
                assert_always(strides[i] == 1 || strides[i] == stencil + 1);
            }
            else
            {
                assert_always(w == strides[i]);
                assert_always(pads[i] == 0);
            }
        }

        bool const use_padding = pads[0] != 0;

        // As halo exchanges with shared tensors is not yet implemented,
        // the spatial domain must be partitioned without sharing or
        // aggregated to the rank-0 process (so that no halo exchange is
        // done).
        for (int i = 0; i < m_num_spatial_dims; ++i)
        {
            if (input.get_distribution().is_shared(i))
            {
                assert_always(input.get_distribution().get_split_shape()[i]
                              == 1);
            }
        }

        // cudnnPoolingBackward requires input and d_input to have the
        // same strides.
        assert_eq(input.get_distribution(), d_input.get_distribution());
        // Similarly, output and d_output must have the same distribution
        assert_eq(output.get_distribution(), output.get_distribution());

        {
            const int_vector dilations(m_num_spatial_dims, 1);
            internal::get_halo_sizes(input,
                                     IntVector(windows),
                                     IntVector(strides),
                                     IntVector(dilations),
                                     m_halo_fwd_send,
                                     m_halo_bwd_send,
                                     m_halo_fwd_recv,
                                     m_halo_bwd_recv,
                                     use_padding);
        }

        GPUDNNBackend::setup_tensor_descriptor(m_input_d,
                                               input,
                                               IntVector(m_halo_fwd_recv),
                                               IntVector(m_halo_bwd_recv));
        util::MPIPrintStreamDebug() << "pooling input desc: " << m_input_d;
        GPUDNNBackend::setup_tensor_descriptor(m_output_d, output, false);

        GPUDNNBackend::setup_tensor_descriptor(
            m_d_input_d, d_input, m_halo_fwd_recv, m_halo_bwd_recv);
        util::MPIPrintStreamDebug() << "pooling d_input desc: " << m_d_input_d;
        GPUDNNBackend::setup_tensor_descriptor(m_d_output_d, d_output, false);

        m_mode =
            pooling::get_pooling_mode(mode, m_be.deterministic());

        // When a dimension is split, halo region works as padding
        for (auto i = pads.begin(); i != pads.end(); i++)
            if (input.get_distribution()
                    .get_split_shape()[std::distance(pads.begin(), i)]
                > 1)
                *i = 0;

        util::MPIPrintStreamDebug()
            << "pooling pads: " << util::join_array(pads, ", ");

        // pooling descriptor
        setup_pooling_descriptor(
            input, output, windows, pads, strides, m_pooling_d);

        setup_halo_xch(input, d_input);

        setup_boundary_streams(input.get_split_index());
        return;
    }

    // FIXME: Default training=true to maximize backward
    // compatibility.
    template <typename Tensor>
    int forward(typename Tensor::data_type alpha,
                Tensor& input,
                typename Tensor::data_type beta,
                Tensor& output,
                bool const training = true)
    {
        exchange_halo_input(input, m_halo_xch_input);

        // Note that even when the local output is empty, halo exchange
        // must be called as this local process may need to push its data
        // to adjacent processes
        if (output.get_local_size() == 0)
        {
            return 0;
        }

        set_num_samples(output.get_local_shape()[-1]);

        const void* input_ptr =
            input.get_const_base_ptr()
            - input.get_local_offset(IndexVector(m_halo_bwd_recv), true);

        m_be.pooling_forward(m_pooling_d,
                             alpha,
                             m_input_d,
                             input_ptr,
                             beta,
                             m_output_d,
                             output.get_base_ptr(),
                             training);
        return 0;
    }

    template <typename Tensor>
    int backward(typename Tensor::data_type alpha,
                 const Tensor& output,
                 const Tensor& d_output,
                 const Tensor& input,
                 typename Tensor::data_type beta,
                 Tensor& d_input)
    {
        if (d_input.get_local_size() == 0)
        {
            return 0;
        }
        set_num_samples(d_input.get_local_shape()[-1]);

        if (d_output.get_local_size() > 0)
        {
            const void* input_ptr =
                input.get_const_base_ptr()
                - input.get_local_offset(IndexVector(m_halo_bwd_recv), true);
            // Assumes d_input has the same distribution as input
            void* d_input_ptr =
                d_input.get_base_ptr()
                - d_input.get_local_offset(IndexVector(m_halo_bwd_recv), true);

            m_be.pooling_backward(m_pooling_d,
                                  alpha,
                                  m_output_d,
                                  output.get_const_base_ptr(),
                                  m_d_output_d,
                                  d_output.get_const_base_ptr(),
                                  m_input_d,
                                  input_ptr,
                                  beta,
                                  m_d_input_d,
                                  d_input_ptr);
        }
#if 0
        {
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      auto m = d_input.get_data();
      auto p = static_cast<DataType*>(malloc(m.get_size()));
      m.copyout(p);
      auto rank = d_input.get_locale().get_rank();
      std::stringstream file_path;
      file_path << "pool_local_" << rank << ".txt";
      std::ofstream out_file;
      out_file.open(file_path.str(), std::ios::out | std::ios::trunc);
      for (int i = 0; i < m.get_size() / sizeof(DataType); ++i) {
        out_file << p[i] << std::endl;
      }
      out_file.close();
        }
#endif
        exchange_halo_reverse(d_input, m_halo_xch_d_input);

        return 0;
    }

    void set_num_samples(int n)
    {
        if (n != GPUDNNBackend::get_tensor_num_samples(m_input_d))
        {
            util::MPIPrintStreamDebug() << "Setting #sample to " << n;
            GPUDNNBackend::set_tensor_num_samples(m_input_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_output_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_d_input_d, n);
            GPUDNNBackend::set_tensor_num_samples(m_d_output_d, n);
        }
    }

    // Wait for asynchronous tasks
    void wait() { m_be.wait(); }

private:
    DNNBackend<GPUDNNBackend>& m_be;
    const int m_num_dims;
    const int m_num_spatial_dims;
    IntVector m_halo_fwd_send;
    IntVector m_halo_bwd_send;
    IntVector m_halo_fwd_recv;
    IntVector m_halo_bwd_recv;
    GPUDNNBackend::TensorDescriptor_t m_input_d;
    GPUDNNBackend::TensorDescriptor_t m_output_d;
    GPUDNNBackend::TensorDescriptor_t m_d_input_d;
    GPUDNNBackend::TensorDescriptor_t m_d_output_d;
    GPUDNNBackend::PoolingDescriptor_t m_pooling_d;
    GPUDNNBackend::PoolingMode_t m_mode;

    HaloExchangeMethod m_halo_xch_method;
    using HaloExchange =
        tensor::HaloExchange<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
    using HaloExchangeMPI = tensor::
        HaloExchangeMPI<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
    using HaloExchangeAL = tensor::
        HaloExchangeAL<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
#ifdef DISTCONV_HAS_P2P
    using HaloExchangeP2P = tensor::
        HaloExchangeP2P<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
    using HaloExchangeHybrid = tensor::
        HaloExchangeHybrid<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
    using HaloExchangeNVSHMEM = tensor::
        HaloExchangeNVSHMEM<DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
#ifdef DISTCONV_HAS_CUDA_GRAPH
    using HaloExchangeNVSHMEMGraph =
        tensor::HaloExchangeNVSHMEMGraph<DataType,
                                         tensor::CUDAAllocator,
                                         Al::NCCLBackend>;
#endif // DISTCONV_HAS_CUDA_GRAPH
    using HaloExchangeNVSHMEMDirect =
        tensor::HaloExchangeNVSHMEMDirect<DataType,
                                          tensor::CUDAAllocator,
                                          Al::NCCLBackend>;
    using HaloExchangeNVSHMEMFusedNotify =
        tensor::HaloExchangeNVSHMEMFusedNotify<DataType,
                                               tensor::CUDAAllocator,
                                               Al::NCCLBackend>;
#endif // DISTCONV_HAS_NVSHMEM
    std::unique_ptr<HaloExchange> m_halo_xch_input;
    std::unique_ptr<HaloExchange> m_halo_xch_d_input;
    BoundaryAttributesV<std::shared_ptr<Al::NCCLBackend::comm_type>>
        m_boundary_comms;

    template <typename Tensor>
    void setup_pooling_descriptor(const Tensor& input,
                                  const Tensor& output,
                                  int_vector windows,
                                  int_vector pads,
                                  int_vector strides,
                                  GPUDNNBackend::PoolingDescriptor_t& pool_d)
    {
        GPUDNNBackend::setup_pooling_descriptor(pool_d,
                                                m_mode,
                                                m_num_spatial_dims,
                                                util::reverse(windows).data(),
                                                util::reverse(pads).data(),
                                                util::reverse(strides).data());
    }

    void bp_accumulate_sum(
        tensor::Tensor<DataType, tensor::LocaleMPI, tensor::CUDAAllocator>&
            tensor,
        IndexVector const& dst,
        IndexVector const& src,
        tensor::Shape const& shape);

    template <typename Allocator>
    void setup_halo_xch(tensor::Tensor<DataType, LocaleMPI, Allocator>& input,
                        tensor::Tensor<DataType, LocaleMPI, Allocator>& d_input)
    {
        switch (m_halo_xch_method)
        {
        case HaloExchangeMethod::MPI:
            util::MPIRootPrintStreamDebug() << "Using MPI in halo exchange";
            m_halo_xch_input.reset(new HaloExchangeMPI(input));
            m_halo_xch_d_input.reset(new HaloExchangeMPI(d_input));
            break;
        case HaloExchangeMethod::AL:
            util::MPIRootPrintStreamDebug() << "Using AL in halo exchange";
            m_halo_xch_input.reset(new HaloExchangeAL(input));
            m_halo_xch_d_input.reset(new HaloExchangeAL(d_input));
            break;
#ifdef DISTCONV_HAS_P2P
        case HaloExchangeMethod::P2P:
            util::MPIRootPrintStreamDebug() << "Using P2P in halo exchange";
            m_halo_xch_input.reset(new HaloExchangeP2P(input, m_be.get_p2p()));
            m_halo_xch_d_input.reset(
                new HaloExchangeP2P(d_input, m_be.get_p2p()));
            break;
        case HaloExchangeMethod::HYBRID:
            util::MPIRootPrintStreamDebug()
                << "Using hybrid of AL and P2P in halo exchange";
            m_halo_xch_input.reset(
                new HaloExchangeHybrid(input, m_be.get_p2p()));
            m_halo_xch_d_input.reset(
                new HaloExchangeHybrid(d_input, m_be.get_p2p()));
            break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
        case HaloExchangeMethod::NVSHMEM:
            m_halo_xch_input.reset(new HaloExchangeNVSHMEM(input));
            m_halo_xch_d_input.reset(new HaloExchangeNVSHMEM(d_input));
            break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
        case HaloExchangeMethod::NVSHMEM_GRAPH:
            m_halo_xch_input.reset(new HaloExchangeNVSHMEMGraph(input));
            m_halo_xch_d_input.reset(new HaloExchangeNVSHMEMGraph(d_input));
            break;
#endif // DISTCONV_HAS_CUDA_GRAPH
        case HaloExchangeMethod::NVSHMEM_DIRECT:
            m_halo_xch_input.reset(new HaloExchangeNVSHMEMDirect(input));
            m_halo_xch_d_input.reset(new HaloExchangeNVSHMEMDirect(d_input));
            break;
        case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
            m_halo_xch_input.reset(new HaloExchangeNVSHMEMFusedNotify(input));
            m_halo_xch_d_input.reset(
                new HaloExchangeNVSHMEMFusedNotify(d_input));
            break;
#endif // DISTCONV_HAS_NVSHMEM
        default:
            util::MPIPrintStreamError()
                << "Invalid halo exchange method: " << m_halo_xch_method;
            std::abort();
        }
    }

    template <typename Allocator>
    void
    exchange_halo_input(tensor::Tensor<DataType, LocaleMPI, Allocator>& tensor,
                        std::unique_ptr<HaloExchange>& xch)
    {
        if (m_be.profiling())
            GPU_PROFILE_RANGE_PUSH("pooling/exchange_halo");
        assert_always(xch != nullptr);
        xch->exchange(
            m_boundary_comms, m_be.get_stream(), false, true, false, false);
        if (m_be.profiling())
            GPU_PROFILE_RANGE_POP();
    }

    template <typename Allocator>
    void exchange_halo_reverse(
        tensor::Tensor<DataType, LocaleMPI, Allocator>& tensor,
        std::unique_ptr<HaloExchange>& xch)
    {
        if (m_be.profiling())
            GPU_PROFILE_RANGE_PUSH("pooling/exchange_halo_rev");
        assert_always(xch != nullptr);
        xch->exchange(m_halo_fwd_recv,
                      m_halo_fwd_send,
                      m_halo_bwd_recv,
                      m_halo_bwd_send,
                      m_boundary_comms,
                      m_be.get_stream(),
                      true,
                      true,
                      true,
                      false,
                      tensor::HaloExchangeAccumOp::SUM);
        if (m_be.profiling())
            GPU_PROFILE_RANGE_POP();
    }

    void setup_boundary_streams(const IndexVector& split_idx)
    {
        apply_to_spatial_sides(m_num_dims, [this](int i, Side side) {
            int idx = get_boundary_stream_index(i, side);
            m_boundary_comms(i, side) = m_be.get_internal_comm(idx);
        });
        for (int i = 0; i < m_num_spatial_dims; ++i)
        {
            if (split_idx[i] % 2)
                std::swap(m_boundary_comms(i, LHS), m_boundary_comms(i, RHS));
        }
    }

    int get_boundary_stream_index(int dim, Side side)
    {
        return dim * 2 + (side == LHS ? 0 : 1);
    }
};

} // namespace distconv
