#pragma once

#include "distconv/util/util.hpp"
#include "distconv/cudnn/backend.hpp"
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda_mpi.hpp"
#include "distconv/tensor/halo_exchange_cuda_al.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/halo_exchange_cuda_p2p.hpp"
#include "distconv/tensor/halo_exchange_cuda_hybrid.hpp"
#endif // DISTCONV_HAS_P2P

namespace distconv {
namespace cudnn {

static inline cudnnPoolingMode_t get_cudnn_pooling_mode(
    const std::string &name, bool deterministic) {
  if (name == "MAX") {
    // This does not seem to be necessary. It's not clear what the
    // difference of the two algorithms is.
    if (deterministic) {
      return CUDNN_POOLING_MAX_DETERMINISTIC;
    } else {
      return CUDNN_POOLING_MAX;
    }
  } else if (name == "AVERAGE") {
    return CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
  } else if (name == "AVERAGE_NO_PAD") {
    return CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  } else {
    util::PrintStreamError() << "No matching pooling mode found for CUDNN: "
                             << name << "\n";
    std::abort();
  }
}

} // namespace cudnn

template <int ND, typename DataType>
class Pooling<cudnn::BackendCUDNN, ND, DataType> {
  using LocaleMPI = tensor::LocaleMPI;
  constexpr static int NSD = ND - 2;

 public:
  Pooling(cudnn::BackendCUDNN &backend,
          HaloExchangeMethod method):
      m_be(backend), m_halo_xch_method(method) {
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreatePoolingDescriptor(&m_pooling_d));
  }

  ~Pooling() {
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyPoolingDescriptor(m_pooling_d));
  }

  Pooling<cudnn::BackendCUDNN, ND, DataType> operator=(
      const Pooling<cudnn::BackendCUDNN, ND, DataType> &x) {
    assert_always(&m_be == &x.m_be);
    cudnn::copy_tensor_descriptor(m_input_d, x.m_input_d);
    cudnn::copy_tensor_descriptor(m_output_d, x.m_output_d);
    cudnn::copy_tensor_descriptor(m_d_input_d, x.m_d_input_d);
    cudnn::copy_tensor_descriptor(m_d_output_d, x.m_d_output_d);
    cudnn::copy_pooling_descriptor(m_pooling_d, x.m_pooling_d);
    return *this;
  }

  template <typename Tensor>
  void setup(Tensor &input,
             Tensor &output,
             Tensor &d_input,
             Tensor &d_output,
             int_vector windows,
             int_vector pads,
             int_vector strides,
             const std::string &mode) {
    {
      // All of the dimensions must be the same
      assert_eq((unsigned int) NSD, windows.size());
      assert_eq((unsigned int) NSD, pads.size());
      assert_eq((unsigned int) NSD, strides.size());
    }

    // TODO: asymmetric not supported
    assert_always(util::is_all_elements_equal(windows));
    assert_always(util::is_all_elements_equal(pads));
    assert_always(util::is_all_elements_equal(strides));

    // TODO: only stencil-like windows are supported
    int_vector stencils;
    for(auto i = windows.begin(); i != windows.end(); i++) {
      assert_eq(*i % 2, 1);
      stencils.push_back((*i - 1) / 2);
    }

    // Padding must be zero or match with the window size
    for(auto i = pads.begin(); i != pads.end(); i++) {
      assert_always(*i == 0 || *i == stencils[std::distance(pads.begin(), i)]);
    }
    bool use_padding = pads[0] != 0;

    // TODO: stride limitation
    for(auto i = strides.begin(); i != strides.end(); i++) {
      assert_always(*i == 1 || *i == stencils[std::distance(strides.begin(), i)] + 1);
    }

    // As halo exchanges with shared tensors is not yet implemented,
    // the spatial domain must be partitioned without sharing or
    // aggregated to the rank-0 process (so that no halo exchange is
    // done).
    for (int i = 0; i < NSD; ++i) {
      if (input.get_distribution().is_shared(i)) {
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
      const int_vector dilations(NSD, 1);
      internal::get_halo_sizes(input,
                               IntVector(windows),
                               IntVector(strides),
                               IntVector(dilations),
                               m_halo_fwd_send, m_halo_bwd_send,
                               m_halo_fwd_recv, m_halo_bwd_recv,
                               use_padding);
    }

    cudnn::setup_tensor_descriptor(m_input_d, input,
                                   IntVector(m_halo_fwd_recv),
                                   IntVector(m_halo_bwd_recv));
    util::MPIPrintStreamDebug()
        << "pooling input desc: " << m_input_d;
    cudnn::setup_tensor_descriptor(m_output_d, output, false);

    cudnn::setup_tensor_descriptor(m_d_input_d, d_input,
                                   m_halo_fwd_recv, m_halo_bwd_recv);
    util::MPIPrintStreamDebug()
        << "pooling d_input desc: " << m_d_input_d;
    cudnn::setup_tensor_descriptor(m_d_output_d, d_output, false);

    m_mode = cudnn::get_cudnn_pooling_mode(
        mode, m_be.get_options().m_deterministic);

    // When a dimension is split, halo region works as padding
    for(auto i = pads.begin(); i != pads.end(); i++)
      if (input.get_distribution().get_split_shape()[std::distance(pads.begin(), i)] > 1)
        *i = 0;

    util::MPIPrintStreamDebug()
        << "pooling pads: " << util::join_array(pads, ", ");

    // pooling descriptor
    setup_pooling_descriptor(input, output,
                             windows,
                             pads,
                             strides,
                             m_pooling_d);

    setup_halo_xch(input, d_input);

    setup_boundary_streams(input.get_split_index());
    return;
  }

  template <typename Tensor>
  int forward(
      typename Tensor::data_type alpha,
      Tensor &input,
      typename Tensor::data_type beta,
      Tensor &output) {

    exchange_halo_input(input, m_halo_xch_input);

    // Note that even when the local output is empty, halo exchange
    // must be called as this local process may need to push its data
    // to adjacent processes
    if (output.get_local_size() == 0) {
      return 0;
    }

    set_num_samples(output.get_local_shape()[-1]);


    const void *input_ptr = input.get_const_base_ptr()
        - input.get_local_offset(IndexVector(m_halo_bwd_recv), true);

    DISTCONV_CHECK_CUDNN(cudnnPoolingForward(
        m_be.get_handle(), m_pooling_d,
        &alpha, m_input_d,
        input_ptr,
        &beta, m_output_d, output.get_base_ptr()));

    return 0;
  }

  template <typename Tensor>
  int backward(
      typename Tensor::data_type alpha,
      const Tensor &output,
      const Tensor &d_output,
      const Tensor &input,
      typename Tensor::data_type beta,
      Tensor &d_input) {

    if (d_input.get_local_size() == 0) {
      return 0;
    }
    set_num_samples(d_input.get_local_shape()[-1]);

    if (d_output.get_local_size() > 0) {

      const void *input_ptr = input.get_const_base_ptr() -
          input.get_local_offset(IndexVector(m_halo_bwd_recv), true);
      // Assumes d_input has the same distribution as input
      void *d_input_ptr = d_input.get_base_ptr() -
          d_input.get_local_offset(IndexVector(m_halo_bwd_recv), true);

      cudnnStatus_t status = cudnnPoolingBackward(
          m_be.get_handle(), m_pooling_d,
          &alpha, m_output_d, output.get_const_base_ptr(),
          m_d_output_d, d_output.get_const_base_ptr(),
          m_input_d, input_ptr,
          &beta, m_d_input_d, d_input_ptr);
      if (status != CUDNN_STATUS_SUCCESS) {
        util::MPIPrintStreamError()
            << "cuDNN error: " << cudnnGetErrorString(status) << "\n"
            << "Error at " << __FILE__ << ":" << __LINE__;
        if (status == CUDNN_STATUS_BAD_PARAM) {
          util::MPIPrintStreamError()
              << "Parameters: "
              << "output_d: " << m_output_d
              << ", output: " << output.get_const_base_ptr()
              << ", d_output_d: " << m_d_output_d
              << ", d_output: " << d_output.get_const_buffer()
              << ", input_d: " << m_input_d
              << ", input: " << input_ptr
              << ", d_input_d: " << m_d_input_d
              << ", d_input: " << d_input_ptr;
        }
        cudaDeviceReset();
        abort();
      }
    }

    {
#if 0
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
#endif
    }
    exchange_halo_reverse(d_input, m_halo_xch_d_input);

    return 0;
  }

  void set_num_samples(int n) {
    if (n != cudnn::get_tensor_num_samples(m_input_d)) {
      util::MPIPrintStreamDebug() << "Setting #sample to " << n;
      cudnn::set_tensor_num_samples(m_input_d, n);
      cudnn::set_tensor_num_samples(m_output_d, n);
      cudnn::set_tensor_num_samples(m_d_input_d, n);
      cudnn::set_tensor_num_samples(m_d_output_d, n);
    }
  }

  // Wait for asynchronous tasks
  void wait() {
    m_be.wait();
  }
 protected:
  cudnn::BackendCUDNN &m_be;
  IntVector m_halo_fwd_send;
  IntVector m_halo_bwd_send;
  IntVector m_halo_fwd_recv;
  IntVector m_halo_bwd_recv;
  cudnnTensorDescriptor_t m_input_d;
  cudnnTensorDescriptor_t m_output_d;
  cudnnTensorDescriptor_t m_d_input_d;
  cudnnTensorDescriptor_t m_d_output_d;
  cudnnPoolingDescriptor_t m_pooling_d;
  cudnnPoolingMode_t m_mode;

  HaloExchangeMethod m_halo_xch_method;
  using HaloExchange = tensor::HaloExchange<DataType,
                                            tensor::CUDAAllocator,
                                            Al::MPICUDABackend>;
  using HaloExchangeMPI = tensor::HaloExchangeMPI<DataType,
                                                  tensor::CUDAAllocator,
                                                  Al::MPICUDABackend>;
  using HaloExchangeAL = tensor::HaloExchangeAL<DataType,
                                                tensor::CUDAAllocator,
                                                Al::MPICUDABackend>;
#ifdef DISTCONV_HAS_P2P
  using HaloExchangeP2P = tensor::HaloExchangeP2P<DataType,
                                                  tensor::CUDAAllocator,
                                                  Al::MPICUDABackend>;
  using HaloExchangeHybrid = tensor::HaloExchangeHybrid<DataType,
                                                        tensor::CUDAAllocator,
                                                        Al::MPICUDABackend>;
#endif // DISTCONV_HAS_P2P
  std::unique_ptr<HaloExchange> m_halo_xch_input;
  std::unique_ptr<HaloExchange> m_halo_xch_d_input;
  BoundaryAttributesV<std::shared_ptr<Al::MPICUDABackend::comm_type>> m_boundary_comms;

  template <typename Tensor>
  void setup_pooling_descriptor(const Tensor &input,
                                const Tensor &output,
                                int_vector windows,
                                int_vector pads,
                                int_vector strides,
                                cudnnPoolingDescriptor_t &pool_d) {
    cudnnNanPropagation_t max_pooling_nan_opt = CUDNN_PROPAGATE_NAN;
    DISTCONV_CHECK_CUDNN(
        cudnnSetPoolingNdDescriptor(pool_d, m_mode, max_pooling_nan_opt,
                                    NSD,
                                    util::reverse(windows).data(),
                                    util::reverse(pads).data(),
                                    util::reverse(strides).data()));
  }

  template <typename Tensor>
  void bp_accumulate_sum(Tensor &tensor,
                         const tensor::Array<ND> dst,
                         const tensor::Array<ND> src,
                         const tensor::Array<ND> shape);

  template <typename Allocator>
  void setup_halo_xch(tensor::Tensor<DataType, LocaleMPI,
                      Allocator> &input,
                      tensor::Tensor<DataType, LocaleMPI,
                      Allocator> &d_input) {
    if (m_halo_xch_method == HaloExchangeMethod::MPI) {
      util::MPIRootPrintStreamDebug() << "Using MPI in halo exchange";
      m_halo_xch_input.reset(new HaloExchangeMPI(input));
      m_halo_xch_d_input.reset(new HaloExchangeMPI(d_input));
    } else if (m_halo_xch_method == HaloExchangeMethod::AL) {
      util::MPIRootPrintStreamDebug() << "Using AL in halo exchange";
      m_halo_xch_input.reset(new HaloExchangeAL(input));
      m_halo_xch_d_input.reset(new HaloExchangeAL(d_input));
#ifdef DISTCONV_HAS_P2P
    } else if (m_halo_xch_method == HaloExchangeMethod::P2P) {
      util::MPIRootPrintStreamDebug() << "Using P2P in halo exchange";
      m_halo_xch_input.reset(new HaloExchangeP2P(input, m_be.get_p2p()));
      m_halo_xch_d_input.reset(new HaloExchangeP2P(d_input, m_be.get_p2p()));
    } else if (m_halo_xch_method == HaloExchangeMethod::HYBRID) {
      util::MPIRootPrintStreamDebug() << "Using hybrid of AL and P2P in halo exchange";
      m_halo_xch_input.reset(new HaloExchangeHybrid(input, m_be.get_p2p()));
      m_halo_xch_d_input.reset(new HaloExchangeHybrid(d_input, m_be.get_p2p()));
#endif // DISTCONV_HAS_P2P
    } else {
      util::MPIPrintStreamError() << "Invalid halo exchange method: "
                                  << m_halo_xch_method;
      std::abort();
    }
  }

  template <typename Allocator>
  void exchange_halo_input(tensor::Tensor<DataType, LocaleMPI,
                           Allocator> &tensor,
                           std::unique_ptr<HaloExchange> &xch) {
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePushA("pooling/exchange_halo");
    }
    assert_always(xch != nullptr);
    xch->exchange(m_boundary_comms, m_be.get_stream(),
                  false, true, false, false);
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePop();
    }
  }

  template <typename Allocator>
  void exchange_halo_reverse(tensor::Tensor<DataType, LocaleMPI,
                             Allocator> &tensor,
                             std::unique_ptr<HaloExchange> &xch) {
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePushA("pooling/exchange_halo_rev");
    }
    assert_always(xch != nullptr);
    xch->exchange(m_halo_fwd_recv,
                  m_halo_fwd_send,
                  m_halo_bwd_recv,
                  m_halo_bwd_send,
                  m_boundary_comms, m_be.get_stream(),
                  true, true, true, false,
                  tensor::HaloExchangeAccumOp::SUM);
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePop();
    }
  }

  void setup_boundary_streams(const tensor::Array<ND> &split_idx) {
    apply_to_spatial_sides<ND>([this](int i, Side side) {
                                 int idx = get_boundary_stream_index(i, side);
                                 m_boundary_comms(i, side) =
                                     m_be.get_internal_al_mpi_cuda_comm(idx);
                               });
    for (int i = 0; i < NSD; ++i) {
      if (split_idx[i] % 2) {
        std::swap(m_boundary_comms(i, LHS), m_boundary_comms(i, RHS));
      }
    }
  }

  int get_boundary_stream_index(int dim, Side side) {
    return dim * 2 + (side == LHS ? 0 : 1);
  }

};

} // namespace distconv
