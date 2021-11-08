#pragma once

#include "distconv/distconv.hpp"
#include "distconv/runtime_cuda.hpp"
#include "distconv/cudnn/backend.hpp"
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_exchange_cuda_mpi.hpp"
#include "distconv/tensor/halo_exchange_cuda_al.hpp"
#ifdef DISTCONV_HAS_P2P
#include "distconv/tensor/halo_exchange_cuda_p2p.hpp"
#include "distconv/tensor/halo_exchange_cuda_hybrid.hpp"
#endif // DISTCONV_HAS_P2P
#include "distconv/tensor/channel_exchange.hpp"
#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/tensor/halo_exchange_cuda_nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM
#include "distconv/util/util_cuda.hpp"
#include "distconv/util/util_cudnn.hpp"

#include "Al.hpp"

#include <memory>

namespace distconv {

template <typename DataType>
class Convolution<cudnn::BackendCUDNN, DataType> {
  using LocaleMPI = tensor::LocaleMPI;

 public:
  Convolution(cudnn::BackendCUDNN &backend,
              int num_dims,
              HaloExchangeMethod method,
              bool enable_overlap, bool enable_profiling,
              ChannelParallelismAlgorithm chanfilt_algo):
      m_be(backend), m_num_dims(num_dims), m_num_spatial_dims(num_dims - 2),
      m_skip_bp_data(false),
      m_ws_size_fwd(0), m_ws_size_bwd_data(0),
      m_ws_size_bwd_filter(0), m_ws_size_fwd_boundaries(0, 0),
      m_halo_xch_method(method),
      m_overlap_halo_exchange_fwd(enable_overlap),
      m_overlap_halo_exchange_bwd(enable_overlap),
      m_enable_profiling(enable_profiling),
      m_chanfilt_algo(chanfilt_algo) {
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_no_halo_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_filter_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_bias_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateFilterDescriptor(&m_d_filter_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_output_no_halo_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_bias_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_conv_fwd_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_conv_bwd_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&m_conv_bwd_filter_d));

    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_interior_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_interior_d));
    apply_to_spatial_sides(m_num_dims, [this](int i, Side side) {
                                 DISTCONV_CHECK_CUDNN(
                                     cudnnCreateTensorDescriptor(&m_input_boundaries_d(i, side)));
                                 DISTCONV_CHECK_CUDNN(
                                     cudnnCreateTensorDescriptor(&m_output_boundaries_d(i, side)));
                               });
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_input_gathered_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_output_all_filters_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_output_gathered_d));
    DISTCONV_CHECK_CUDNN(cudnnCreateTensorDescriptor(&m_d_input_all_channels_d));
    setup_profiling_events();
  }

  Convolution(cudnn::BackendCUDNN &backend,
              int num_dims,
              HaloExchangeMethod method,
              ChannelParallelismAlgorithm chanfilt_algo=ChannelParallelismAlgorithm::AUTO):
      Convolution(backend, num_dims, method,
                  backend.get_options().m_overlap_halo_exchange,
                  backend.get_options().m_enable_profiling,
                  chanfilt_algo) {}

  ~Convolution() {
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_no_halo_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_filter_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_bias_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_input_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyFilterDescriptor(m_d_filter_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_output_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_output_no_halo_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_bias_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_conv_fwd_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_conv_bwd_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(m_conv_bwd_filter_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_input_gathered_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_output_all_filters_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_output_gathered_d));
    DISTCONV_CHECK_CUDNN(cudnnDestroyTensorDescriptor(m_d_input_all_channels_d));
    destroy_profiling_events();
  }

  Convolution<cudnn::BackendCUDNN, DataType> operator=(
      const Convolution<cudnn::BackendCUDNN, DataType> &x) {
    assert_always(&m_be == &x.m_be);
    cudnn::copy_tensor_descriptor(m_input_d, x.m_input_d);
    cudnn::copy_tensor_descriptor(m_input_no_halo_d, x.m_input_no_halo_d);
    cudnn::copy_filter_descriptor(m_filter_d, x.m_filter_d);
    cudnn::copy_tensor_descriptor(m_output_d, x.m_output_d);
    cudnn::copy_tensor_descriptor(m_bias_d, x.m_bias_d);
    cudnn::copy_tensor_descriptor(m_d_input_d, x.m_d_input_d);
    cudnn::copy_filter_descriptor(m_d_filter_d, x.m_d_filter_d);
    cudnn::copy_tensor_descriptor(m_d_output_d, x.m_d_output_d);
    cudnn::copy_tensor_descriptor(m_d_output_no_halo_d, x.m_d_output_no_halo_d);
    cudnn::copy_tensor_descriptor(m_d_bias_d, x.m_d_bias_d);
    cudnn::copy_convolution_descriptor(m_conv_fwd_d, x.m_conv_fwd_d);
    cudnn::copy_convolution_descriptor(m_conv_bwd_d, x.m_conv_bwd_d);
    cudnn::copy_convolution_descriptor(m_conv_bwd_filter_d,
                                       x.m_conv_bwd_filter_d);
    cudnn::copy_tensor_descriptor(m_input_gathered_d, x.m_input_gathered_d);
    cudnn::copy_tensor_descriptor(m_output_all_filters_d, x.m_output_all_filters_d);
    cudnn::copy_tensor_descriptor(m_d_output_gathered_d, x.m_d_output_gathered_d);
    cudnn::copy_tensor_descriptor(m_d_input_all_channels_d, x.m_d_input_all_channels_d);
    m_fwd_algo = x.m_fwd_algo;
    m_fwd_boundary_algos = x.m_fwd_boundary_algos;
    m_bwd_data_algo = x.m_bwd_data_algo;
    m_bwd_filter_algo = x.m_bwd_filter_algo;
    m_ws_size_fwd = x.m_ws_size_fwd;
    m_ws_size_bwd_data = x.m_ws_size_bwd_data;
    m_ws_size_bwd_filter = x.m_ws_size_bwd_filter;
    m_ws_size_fwd_boundaries = x.m_ws_size_fwd_boundaries;
#if 0
    for (int i = 0; i < 2; ++i) {
      m_ddt_fwd[i] = x.m_ddt_fwd[i];
      m_ddt_bwd[i] = x.m_ddt_bwd[i];
    }
#endif
    m_halo_xch_method = x.m_halo_xch_method;
    switch (m_halo_xch_method) {
      case HaloExchangeMethod::MPI:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeMPI(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeMPI(x.m_halo_xch_d_output));
        }
        break;
      case HaloExchangeMethod::AL:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeAL(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeAL(x.m_halo_xch_d_output));
        }
        break;
#ifdef DISTCONV_HAS_P2P
      case HaloExchangeMethod::P2P:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeP2P(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeP2P(x.m_halo_xch_d_output));
        }
        break;
      case HaloExchangeMethod::HYBRID:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeHybrid(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeHybrid(x.m_halo_xch_d_output));
        }
        break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
      case HaloExchangeMethod::NVSHMEM:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeNVSHMEM(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeNVSHMEM(x.m_halo_xch_d_output));
        }
        break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
      case HaloExchangeMethod::NVSHMEM_GRAPH:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeNVSHMEMGraph(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMGraph(x.m_halo_xch_d_output));
        }
        break;
#endif // DISTCONV_HAS_CUDA_GRAPH
      case HaloExchangeMethod::NVSHMEM_DIRECT:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeNVSHMEMDirect(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMDirect(x.m_halo_xch_d_output));
        }
        break;
      case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
        if (x.m_halo_xch_input) {
          m_halo_xch_input.reset(new HaloExchangeNVSHMEMFusedNotify(x.m_halo_xch_input));
        }
        if (x.m_halo_xch_d_output) {
          m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMFusedNotify(x.m_halo_xch_d_output));
        }
        break;
#endif // DISTCONV_HAS_NVSHMEM
      default:
        util::MPIPrintStreamError() << "Invalid halo exchange method: "
                                    << m_halo_xch_method;
        std::abort();
    }
    m_chanfilt_algo = x.m_chanfilt_algo;
    return *this;
  }

  template <typename Allocator>
  void setup(tensor::Tensor<DataType, LocaleMPI,
             Allocator> &input,
             const tensor::Tensor<DataType, LocaleMPI,
             Allocator> &filter,
             const tensor::Tensor<DataType, LocaleMPI,
             Allocator> &output,
             const tensor::Tensor<DataType, LocaleMPI,
             Allocator> &d_input,
             tensor::Tensor<DataType, LocaleMPI,
             Allocator> &d_filter,
             tensor::Tensor<DataType, LocaleMPI,
             Allocator> &d_output,
             const int_vector &pads,
             const int_vector &strides,
             const int_vector &dilations,
             int num_groups,
             const std::string &fwd_algo,
             const std::string &bwd_data_algo,
             const std::string &bwd_filter_algo,
             size_t ws_size,
             bool skip_bp_data=false,
             bool deconv=false) {
    // NVSHMEM-exchange requires all processes join the allocation of
    // halo buffers, so this must be called even the local buffer is
    // empty.
    setup_halo_xch(input, d_output);

    if (input.get_local_size() == 0 ||
        output.get_local_size() == 0) {
      util::MPIPrintStreamInfo() << "Empty tensor detected";
      setup_chanfilt_comms(input, filter);  // Still need to participate in this.
      return;
    }

    select_chanfilt_algorithm(input, filter, output);

    m_skip_bp_data = skip_bp_data;
    m_deconv = deconv;

    std::vector<int> stencil_dims(m_num_spatial_dims, 0);
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      auto window_dim = internal::get_dilated_filter_size(
          (int)filter.get_shape()[i], dilations[i]);
      if (window_dim % 2) {
        stencil_dims[i] = (window_dim - 1) / 2;
      } else {
        // Allow even-shaped filters only when no spatial data
        // dependency exists
        assert_eq(strides[i], window_dim);
        stencil_dims[i] = 0;
      }
    }

    auto p = pads[0];
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      assert_eq(pads[i], p);
      assert_always(pads[i] == stencil_dims[i] || pads[i] == 0);
    }
    bool use_padding = p != 0;

    internal::get_halo_sizes(input, IntVector(filter.get_shape()),
                             IntVector(strides), IntVector(dilations),
                             m_halo_fwd_send, m_halo_bwd_send,
                             m_halo_fwd_recv, m_halo_bwd_recv, use_padding);

    util::MPIPrintStreamDebug()
        << "halo size: " << m_halo_fwd_recv[1] << ", " << m_halo_bwd_recv[1];

    if (m_overlap_halo_exchange_fwd) {
      // Disables fwd overlapping for tensors with small spatial domains
      // as it would be unlikely to be profitable.
      for (int i = 0; i < m_num_spatial_dims; ++i) {
        // TODO: Parameterize the constant "3"?
        if (input.get_local_shape()[i] < (index_t)stencil_dims[i] * 3) {
          util::MPIRootPrintStreamInfo()
              << "Overlapped halo exchange in forward convolution disabled as the spatial domain is small ("
              << input.get_local_shape();
          m_overlap_halo_exchange_fwd = false;
        }
      }
    }

    if (m_overlap_halo_exchange_bwd) {
      // The overlapping with the task parallelism does not correctly
      // communicate diagonal halo points. The number of the
      // partitioning dimensions must be one.
      auto dist = input.get_distribution();
      int num_partitioned_dims = 0;
      for (int i = 0; i < m_num_spatial_dims; ++i) {
        if (dist.get_locale_shape()[i] > 1) ++num_partitioned_dims;
      }
      if (num_partitioned_dims > 1) {
        m_overlap_halo_exchange_bwd = false;
        util::MPIRootPrintStreamInfo()
            << "Overlapped halo exchange in backward convolution disabled"
            << " as the tensor is partitioned in multiple dimensions";
      }
    }

    // Not strictly necessary, but disables overlapping when halo
    // exchange is not necessary
    bool halo_exchange_required = false;
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      if (stencil_dims[i] > 0 && input.get_distribution().get_split_shape()[i] > 1) {
        halo_exchange_required = true;
        break;
      }
    }
    if (!halo_exchange_required) {
      util::MPIPrintStreamDebug() << "Halo exchange not required";
      m_overlap_halo_exchange_fwd = false;
      m_overlap_halo_exchange_bwd = false;
    }

    if (m_overlap_halo_exchange_fwd) {
      util::MPIRootPrintStreamDebug()
          << "Overlapping of halo exchanges in forward convolution enabled";
    }
    if (m_overlap_halo_exchange_bwd) {
      util::MPIRootPrintStreamDebug()
          << "Overlapping of halo exchanges in backward convolution enabled";
    }

    setup_tensor_descriptors(
        input, filter, output, d_input, d_filter,
        d_output, strides, dilations);

    setup_convolution_descriptor(
        input.get_overlap(), filter.get_shape(),
        pads, strides, dilations, num_groups,
        m_conv_fwd_d, m_conv_bwd_d, m_conv_bwd_filter_d);
    util::MPIPrintStreamDebug() << "Convolution fwd desc: "
                                << m_conv_fwd_d
                                << "\n bwd data: "
                                << m_conv_bwd_d
                                << "\n bwd filter: "
                                << m_conv_bwd_filter_d;

    // when 0 is given as workspace size, use 80 % of currently
    // available memory
    // TODO: Parameterize the constant "0.8"?
    if (ws_size == 0) {
      size_t available;
      size_t total;
      DISTCONV_CHECK_CUDA(cudaMemGetInfo(&available, &total));
      ws_size = available * 0.8;
    }

    auto &mempool = internal::RuntimeCUDA::get_device_memory_pool();
    // take the maximum size that does not exceed the request size
    auto actual_ws_size = mempool.get_max_allocatable_size(ws_size);
    // adjust the size with an option value
    actual_ws_size *= m_be.get_options().m_ws_capacity_factor;
    actual_ws_size = mempool.get_max_allocatable_size(actual_ws_size);

    util::MPIRootPrintStreamDebug()
        << "Requested workspace size: " << ws_size
        << " (" << int(ws_size / 1024.0 / 1024.0)
        << " MB), actual size: " << actual_ws_size
        << " (" << int(actual_ws_size / 1024.0 / 1024.0)
        << " MB)";
    setup_algorithms(fwd_algo, bwd_data_algo, bwd_filter_algo,
                     input.get_buffer(), d_filter.get_buffer(),
                     d_output.get_buffer(), actual_ws_size);
    setup_workspace_sizes();
  }

  // Setup of bias gradient should be done by a different function as
  // it is optional
  template <typename Allocator>
  void setup_bias(const tensor::Tensor<DataType, LocaleMPI,
                  Allocator> &bias) {
    cudnn::setup_tensor_descriptor(m_bias_d, bias, false);
    util::MPIPrintStreamDebug() << "bias: " << m_bias_d;
  }

  template <typename Allocator>
  void setup_bias_gradient(
      const tensor::Tensor<DataType, LocaleMPI,
      Allocator> &d_bias) {
    cudnn::setup_tensor_descriptor(m_d_bias_d, d_bias, false);
    util::MPIPrintStreamDebug() << "d_bias: " << m_d_bias_d;
  }

  template <typename Allocator>
  int forward_exchange_halo(
      tensor::Tensor<DataType, LocaleMPI, Allocator> &input) {
    if (input.get_local_size() == 0) {
      util::MPIPrintStreamDebug()
          << "Skipping forward convolution with an empty tensor";
      return 0;
    }
    exchange_halo(input, m_halo_xch_input, m_overlap_halo_exchange_fwd);
    return 0;
  }

  template <typename Allocator>
  int forward(
      DataType alpha,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &input,
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &filter,
      DataType beta,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &output,
      bool skip_halo_exchange=false,
      bool skip_chanfilt_comm=false,
      bool dump_profile=false,
      bool inference=false) {

    if (input.get_local_size() == 0 ||
        filter.get_local_size() == 0 ||
        output.get_local_size() == 0) {
      util::MPIPrintStreamDebug()
          << "Skipping forward convolution with an empty tensor";
      return 0;
    }

    if (m_be.is_nvtx_enabled()) {
      nvtxRangePushA("conv/forward");
    }

    set_num_samples(input.get_local_shape()[-1]);

    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      get_tmp_tensor_buffer(m_output_all_filters_t);
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      // This is retained until back-filter completes unless doing inference.
      get_tmp_tensor_buffer(m_input_gathered_t);
    }

    if (!skip_chanfilt_comm &&
        (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W)) {
      // Allgather to assemble x by channels.
      allgather_chanfilt(input, m_input_gathered_t, true);
    }

    void *ws = internal::RuntimeCUDA::get_device_memory_pool().get(
        m_ws_size_fwd, m_be.get_stream());

    if (ws == nullptr) return -1;

    if (!skip_halo_exchange) forward_exchange_halo(input);

    const void *input_ptr = input.get_const_base_ptr() -
        input.get_local_offset(IndexVector(m_halo_bwd_recv), true);

    if (m_be.is_nvtx_enabled()) {
      nvtxRangePushA("conv/forward/forward");
    }

    if (!m_overlap_halo_exchange_fwd) {
      record_start_comp();
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
        ensure_tensors_conform(
          input, m_output_all_filters_t, filter,
          "stationary-x forward");
        ensure_tensor_descriptors_conform(
          m_input_d, m_output_all_filters_d, m_filter_d,
          "stationary-x forward");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
          m_be.get_handle(), &alpha, m_input_d, input_ptr,
          m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
          m_fwd_algo, ws, m_ws_size_fwd,
          &beta, m_output_all_filters_d,
          m_output_all_filters_t.get_base_ptr()));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
        // TODO: Support halos.
        ensure_tensors_conform(
          m_input_gathered_t, output, filter,
          "stationary-y forward");
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_d, m_filter_d,
          "stationary-y forward");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
          m_be.get_handle(), &alpha,
          m_input_gathered_d, m_input_gathered_t.get_base_ptr(),
          m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
          m_fwd_algo, ws, m_ws_size_fwd,
          &beta, m_output_d, output.get_base_ptr()));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        // TODO: Support halos.
        ensure_tensors_conform(
          m_input_gathered_t, m_output_all_filters_t, filter,
          "stationary-w forward");
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_all_filters_d, m_filter_d,
          "stationary-w forward");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
          m_be.get_handle(), &alpha,
          m_input_gathered_d, m_input_gathered_t.get_base_ptr(),
          m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
          m_fwd_algo, ws, m_ws_size_fwd,
          &beta, m_output_all_filters_d,
          m_output_all_filters_t.get_base_ptr()));
      } else {
        // REFACTORING: Temporary adds deconv only this case
        if (!m_deconv) {
          ensure_tensors_conform(
            input, output, filter,
            "forward");
          ensure_tensor_descriptors_conform(
            m_input_d, m_output_d, m_filter_d,
            "forward");
          DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
              m_be.get_handle(), &alpha, m_input_d, input_ptr,
              m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
              m_fwd_algo, ws, m_ws_size_fwd,
              &beta, m_output_d, output.get_base_ptr()));
        } else {
          DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
              m_be.get_handle(), &alpha,
              m_filter_d, filter.get_const_base_ptr(),
              m_input_d, input_ptr,
              m_conv_fwd_d, m_bwd_data_algo, ws, m_ws_size_fwd,
              &beta, m_output_d, output.get_base_ptr()));
        }
      }
      record_end_comp();
    } else {
      record_start_comp();
      // TODO: Support chanfilt.
      if (m_interior_req) {
        const void *input_interior_ptr = input.get_const_buffer()
            + m_input_interior_offset;
        void *output_interior_ptr = output.get_buffer()
            + m_output_interior_offset;
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
            m_be.get_handle(), &alpha, m_input_interior_d,
            input_interior_ptr,
            m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
            m_fwd_algo, ws, m_ws_size_fwd, &beta,
            m_output_interior_d, output_interior_ptr));
      }
      record_end_comp();
      apply_to_spatial_sides(
          m_num_dims,
          [&](int i, Side side) {
            if (!m_boundary_req(i, side)) return;
            const void *boundary_input_ptr = input.get_const_buffer()
                + m_input_boundary_offsets(i, side);
            void *boundary_output_ptr = output.get_buffer()
                + m_output_boundary_offsets(i, side);
            cudaStream_t st_boundary = get_boundary_stream(i, side);
            void *ws_boundary = internal::RuntimeCUDA::get_device_memory_pool().get(
                m_ws_size_fwd_boundaries(i, side), st_boundary);
            util::MPIPrintStreamDebug()
                << "Launching convolution of boundary at dimension "
                << i << ", side: " << side;
            record_start_boundary(i, side);
            DISTCONV_CHECK_CUDNN(cudnnSetStream(m_be.get_handle(),
                                                st_boundary));
            DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
                m_be.get_handle(), &alpha, m_input_boundaries_d(i, side),
                boundary_input_ptr,
                m_filter_d, filter.get_const_base_ptr(), m_conv_fwd_d,
                m_fwd_boundary_algos(i, side),
                ws_boundary, m_ws_size_fwd_boundaries(i, side),
                &beta, m_output_boundaries_d(i, side), boundary_output_ptr));
            record_end_boundary(i, side);
            internal::RuntimeCUDA::get_device_memory_pool().release(ws_boundary);
            util::wait_stream(st_boundary, m_be.get_stream());
          });
      DISTCONV_CHECK_CUDNN(cudnnSetStream(m_be.get_handle(),
                                          m_be.get_stream()));
    }

    if (!skip_chanfilt_comm &&
        (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W)) {
      // Reduce-scatter to complete the sum over channels.
      reduce_scatter_chanfilt(m_output_all_filters_t, output, false);
    }

    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      release_tmp_tensor_buffer(m_output_all_filters_t);
    }

    if (inference &&
        (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W)) {
      release_tmp_tensor_buffer(m_input_gathered_t);
    }

    internal::RuntimeCUDA::get_device_memory_pool().release(ws);

    if (m_be.is_nvtx_enabled()) {
      m_be.wait();
      nvtxRangePop();
    }

    if (m_be.is_nvtx_enabled()) {
      nvtxRangePop();
    }

    if (dump_profile) dump_profile_statistics(true, true, true);

    return 0;
  }

  template <typename TensorType>
  int apply_bias(
      typename TensorType::data_type alpha,
      const TensorType &bias,
      typename TensorType::data_type beta,
      TensorType &output) {
    if (output.get_local_size() == 0) return 0;

    set_num_samples(output.get_local_shape()[-1]);

    DISTCONV_CHECK_CUDNN(cudnnAddTensor(
        m_be.get_handle(),
        &alpha, m_bias_d, bias.get_const_base_ptr(),
        &beta, m_output_d, output.get_base_ptr()));
    return 0;
  }

  // Start halo exchange for backward data
  template <typename Allocator>
  int backward_data_exchange_halo(
      tensor::Tensor<DataType, LocaleMPI, Allocator> &d_output) {
    if (d_output.get_local_size() == 0) {
      return 0;
    }
    set_num_samples(d_output.get_local_shape()[-1]);
    // When done asynchronously, the halo regions must not be modified
    // yet as they are read by the backward filter convolution, so the
    // skip option must be used
    exchange_halo(d_output, m_halo_xch_d_output, m_overlap_halo_exchange_bwd,
                  m_overlap_halo_exchange_bwd);
    return 0;
  }

  template <typename Allocator>
  int backward_data(
      DataType alpha,
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &filter,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &d_output,
      DataType beta,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &d_input,
      bool skip_halo_exchange=false,
      bool skip_chanfilt_comm=false,
      bool dump_profile=false) {

    if (m_skip_bp_data) {
      util::MPIRootPrintStreamInfo()
          << "backward_data was called despite configured to skip";
      return 0;
    }

    if (filter.get_local_size() == 0 ||
        d_output.get_local_size() == 0 ||
        d_input.get_local_size() == 0) {
      return 0;
    }

    // Handle case where backward_filter was not called.
    if ((m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W) &&
        m_d_output_gathered_t.get_buffer() == nullptr) {
      get_tmp_tensor_buffer(m_d_output_gathered_t);
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      get_tmp_tensor_buffer(m_d_input_all_channels_t);
    }

    if (!m_overlap_halo_exchange_bwd && !skip_halo_exchange) {
      backward_data_exchange_halo(d_output);
    }

    void *ws = internal::RuntimeCUDA::get_device_memory_pool().get(
        m_ws_size_bwd_data, m_be.get_stream());
    if (ws == nullptr) return -1;

    void *d_input_ptr = d_input.get_base_ptr() -
        d_input.get_local_offset(m_halo_bwd_recv, true);

    if (m_overlap_halo_exchange_bwd) {
      unpack_halo(d_output, m_halo_xch_d_output);
    }

    record_start_comp();
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
      ensure_tensors_conform(
        d_input, m_d_output_gathered_t, filter,
        "stationary-x backward-data");
      ensure_tensor_descriptors_conform(
          m_d_input_d, m_d_output_gathered_d, m_filter_d,
          "stationary-x backward-data");
      // Assumes d_output was allgathered by backward_filter.
      DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
        m_be.get_handle(), &alpha, m_filter_d, filter.get_const_base_ptr(),
        m_d_output_gathered_d, m_d_output_gathered_t.get_const_buffer(),
        m_conv_bwd_d, m_bwd_data_algo, ws, m_ws_size_bwd_data,
        &beta, m_d_input_d, d_input_ptr));
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      ensure_tensors_conform(
        m_d_input_all_channels_t, d_output, filter,
        "stationary-y backward-data");
      ensure_tensor_descriptors_conform(
          m_d_input_all_channels_d, m_d_output_d, m_filter_d,
          "stationary-y backward-data");
      // TODO: Handle halos.
      DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
        m_be.get_handle(), &alpha, m_filter_d, filter.get_const_base_ptr(),
        m_d_output_d, d_output.get_const_buffer(),
        m_conv_bwd_d, m_bwd_data_algo, ws, m_ws_size_bwd_data,
        &beta, m_d_input_all_channels_d,
        m_d_input_all_channels_t.get_base_ptr()));
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      ensure_tensors_conform(
        m_d_input_all_channels_t, m_d_output_gathered_t, filter,
        "stationary-w backward-data");
      ensure_tensor_descriptors_conform(
          m_d_input_all_channels_d, m_d_output_gathered_d, m_filter_d,
          "stationary-w backward-data");
      // TODO: Handle halos.
      // Assumes d_output was allgathered by backward_filter.
      DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
        m_be.get_handle(), &alpha, m_filter_d, filter.get_const_base_ptr(),
        m_d_output_gathered_d, m_d_output_gathered_t.get_const_buffer(),
        m_conv_bwd_d, m_bwd_data_algo, ws, m_ws_size_bwd_data,
        &beta, m_d_input_all_channels_d,
        m_d_input_all_channels_t.get_base_ptr()));
    } else {
      if (!m_deconv) {
        ensure_tensors_conform(
          d_input, d_output, filter,
          "backward-data");
        ensure_tensor_descriptors_conform(
          m_d_input_d, m_d_output_d, m_filter_d,
          "backward-data");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardData(
            m_be.get_handle(), &alpha, m_filter_d, filter.get_const_base_ptr(),
            m_d_output_d, d_output.get_const_buffer(),
            m_conv_bwd_d, m_bwd_data_algo, ws, m_ws_size_bwd_data,
            &beta, m_d_input_d, d_input_ptr));
      } else {
        DISTCONV_CHECK_CUDNN(cudnnConvolutionForward(
            m_be.get_handle(), &alpha, m_d_output_d, d_output.get_const_buffer(),
            m_filter_d, filter.get_const_base_ptr(),
            m_conv_bwd_d, m_fwd_algo, ws, m_ws_size_bwd_data,
            &beta, m_d_input_d, d_input_ptr));
      }
    }
    if (!skip_chanfilt_comm &&
        (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W)) {
      // Reduce-scatter to complete the sum over filters.
      reduce_scatter_chanfilt(m_d_input_all_channels_t, d_input, true);
    }
    record_end_comp();
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      release_tmp_tensor_buffer(m_d_output_gathered_t);
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      release_tmp_tensor_buffer(m_d_input_all_channels_t);
    }
    internal::RuntimeCUDA::get_device_memory_pool().release(ws);

    if (dump_profile) dump_profile_statistics(true, false, false);
    return 0;
  }

  template <typename Allocator>
  int backward_filter(
      DataType alpha,
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &input,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &d_output,
      DataType beta,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &d_filter,
      bool reduce=true,
      bool skip_chanfilt_comm=false,
      bool dump_profile=false) {

    if (input.get_local_size() == 0 ||
        d_output.get_local_size() == 0 ||
        d_filter.get_local_size() == 0 ||
        !input.is_split_root()) {
      d_filter.zero(m_be.get_stream());
      if (reduce) allreduce_gradients(d_filter);
      return 0;
    }

    set_num_samples(input.get_local_shape()[-1]);

    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      // This is released by backward data.
      if (m_d_output_gathered_t.get_buffer() == nullptr) {
        get_tmp_tensor_buffer(m_d_output_gathered_t);
      }
    }

    if (!skip_chanfilt_comm &&
        (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W)) {
      // TODO: backward_data relies on this.
      // Allgather to assemble dL/dy by filters.
      allgather_chanfilt(d_output, m_d_output_gathered_t, false);
    }

    // Handle case where forward was not called.
    if ((m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
         m_chanfilt_algo == ChannelParallelismAlgorithm::W) &&
        m_input_gathered_t.get_buffer() == nullptr) {
      get_tmp_tensor_buffer(m_input_gathered_t);
    }

    // Is there any case where d_filter is empty?
    assert_always(d_filter.get_local_size() != 0);

    record_start_comp();

    if (input.get_local_size() == 0 ||
        d_output.get_local_size() == 0 ||
        !input.is_split_root()) {
      // This process still needs to join Allreduce
      d_filter.zero(m_be.get_stream());
    } else {
      void *ws = internal::RuntimeCUDA::get_device_memory_pool().get(
          m_ws_size_bwd_filter, m_be.get_stream());
      if (ws == nullptr) return -1;

      // Zero-clear the halo region of the d_output
      for (int dim = 0; dim < m_num_spatial_dims; ++dim) {
        const auto &dist = d_output.get_distribution();
        if (dist.is_distributed(dim) && dist.get_locale_shape()[dim] > 1 &&
            dist.get_overlap(dim) > 0) {
          d_output.clear_halo(dim, m_be.get_stream());
        }
      }

      const void *input_ptr = input.get_const_base_ptr() -
          input.get_local_offset(m_halo_bwd_recv, true);
      assert_always(input_ptr != nullptr);
      assert_always(d_output.get_const_buffer() != nullptr);
      assert_always(d_filter.get_buffer() != nullptr);
      util::MPIPrintStreamDebug() << "Running Bp filter";
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
        ensure_tensors_conform(
          input, m_d_output_gathered_t, d_filter,
          "stationary-x backward-filter");
        ensure_tensor_descriptors_conform(
          m_input_d, m_d_output_gathered_d, m_filter_d,
          "stationary-x backward-filter");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
          m_be.get_handle(), &alpha, m_input_d,
          input_ptr,
          m_d_output_gathered_d, m_d_output_gathered_t.get_const_buffer(),
          m_conv_bwd_filter_d, m_bwd_filter_algo, ws, m_ws_size_bwd_filter,
          &beta, m_d_filter_d, d_filter.get_buffer()));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
        ensure_tensors_conform(
          m_input_gathered_t, d_output, d_filter,
          "stationary-y backward-filter");
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_d_output_d, m_filter_d,
          "stationary-y backward-filter");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
          m_be.get_handle(), &alpha, m_input_gathered_d,
          m_input_gathered_t.get_base_ptr(),
          m_d_output_d, d_output.get_const_buffer(),
          m_conv_bwd_filter_d, m_bwd_filter_algo, ws, m_ws_size_bwd_filter,
          &beta, m_d_filter_d, d_filter.get_buffer()));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        ensure_tensors_conform(
          m_input_gathered_t, m_d_output_gathered_t, d_filter,
          "stationary-w backward-filter");
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_d_output_gathered_d, m_filter_d,
          "stationary-w backward-filter");
        DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
          m_be.get_handle(), &alpha, m_input_gathered_d,
          m_input_gathered_t.get_base_ptr(),
          m_d_output_gathered_d, m_d_output_gathered_t.get_const_buffer(),
          m_conv_bwd_filter_d, m_bwd_filter_algo, ws, m_ws_size_bwd_filter,
          &beta, m_d_filter_d, d_filter.get_buffer()));
      } else {
        if (!m_deconv) {
          ensure_tensors_conform(
            input, d_output, d_filter,
            "backward-filter");
          ensure_tensor_descriptors_conform(
            m_input_d, m_d_output_d, m_filter_d,
            "backward-filter");
          DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
              m_be.get_handle(), &alpha, m_input_d,
              input_ptr,
              m_d_output_d, d_output.get_const_buffer(),
              m_conv_bwd_filter_d, m_bwd_filter_algo, ws, m_ws_size_bwd_filter,
              &beta, m_d_filter_d, d_filter.get_buffer()));
        } else {
          DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardFilter(
              m_be.get_handle(), &alpha,
              m_d_output_d, d_output.get_const_buffer(),
              m_input_d, input_ptr,
              m_conv_bwd_filter_d, m_bwd_filter_algo, ws, m_ws_size_bwd_filter,
              &beta, m_d_filter_d, d_filter.get_buffer()));
        }
      }

      internal::RuntimeCUDA::get_device_memory_pool().release(ws);

      util::MPIPrintStreamDebug() << "Bp filter done";
    }

    if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      release_tmp_tensor_buffer(m_input_gathered_t);
    }

    record_end_comp();

    if (reduce) allreduce_gradients(d_filter);

    if (dump_profile) dump_profile_statistics(false, false, false);

    return 0;
  }

  template <typename Allocator>
  int backward_bias(
      DataType alpha,
      const tensor::Tensor<DataType, LocaleMPI, Allocator> &d_output,
      DataType beta,
      tensor::Tensor<DataType, LocaleMPI, Allocator> &bias_gradient,
      bool reduce=true,
      bool dump_profile=false) {

    if (d_output.get_local_size() == 0) {
      bias_gradient.zero(m_be.get_stream());
      if (reduce) allreduce_gradients(bias_gradient);
      return 0;
    }

    set_num_samples(d_output.get_local_shape()[-1]);
    record_start_comp();
    DISTCONV_CHECK_CUDNN(cudnnConvolutionBackwardBias(
        m_be.get_handle(),
        &alpha, m_d_output_no_halo_d,
        d_output.get_const_base_ptr(),
        &beta, m_d_bias_d, bias_gradient.get_base_ptr()));
    record_end_comp();

    if (reduce) allreduce_gradients(bias_gradient);

    if (dump_profile) dump_profile_statistics(false, false, false);
    return 0;
  }

  // Wait for asynchronous tasks
  void wait() {
    m_be.wait();
  }

  void set_num_samples(int n) {
    assert_ne(n, 0);
    // Set all the tensor descriptors. No need to adjust MPI
    // datatypes, although MPI transfers will incur extra movement
    if (n != cudnn::get_tensor_num_samples(m_input_d)) {
      util::MPIPrintStreamDebug()
          << "Setting #sample to " << n
          << " from " << cudnn::get_tensor_num_samples(m_input_d);
      cudnn::set_tensor_num_samples(m_input_d, n);
      cudnn::set_tensor_num_samples(m_input_no_halo_d, n);
      cudnn::set_tensor_num_samples(m_output_d, n);
      if (!m_skip_bp_data) {
        cudnn::set_tensor_num_samples(m_d_input_d, n);
      }
      cudnn::set_tensor_num_samples(m_d_output_d, n);
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
          m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        cudnn::set_tensor_num_samples(m_output_all_filters_d, n);
        cudnn::set_tensor_num_samples(m_d_output_gathered_d, n);
      }
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y ||
          m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        cudnn::set_tensor_num_samples(m_input_gathered_d, n);
        cudnn::set_tensor_num_samples(m_d_input_all_channels_d, n);
      }
      setup_workspace_sizes();
    }
  }

  bool is_overlap_fwd_halo_exchange_enabled() const {
    return m_overlap_halo_exchange_fwd;
  }

  bool is_overlap_bwd_halo_exchange_enabled() const {
    return m_overlap_halo_exchange_bwd;
  }

  cudnnConvolutionFwdAlgo_t get_fwd_algo() const {
    return m_fwd_algo;
  }

  cudnnConvolutionBwdDataAlgo_t get_bwd_data_algo() const {
    return m_bwd_data_algo;
  }

  cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algo() const {
    return m_bwd_filter_algo;
  }

 protected:
  cudnn::BackendCUDNN &m_be;
  const int m_num_dims;
  const int m_num_spatial_dims;
  bool m_skip_bp_data;
  bool m_deconv;
  cudnnTensorDescriptor_t m_input_d;
  cudnnTensorDescriptor_t m_input_no_halo_d;
  cudnnFilterDescriptor_t m_filter_d;
  cudnnTensorDescriptor_t m_output_d;
  cudnnTensorDescriptor_t m_d_input_d;
  cudnnFilterDescriptor_t m_d_filter_d;
  cudnnTensorDescriptor_t m_d_output_d;
  cudnnTensorDescriptor_t m_d_output_no_halo_d;
  cudnnTensorDescriptor_t m_bias_d;
  cudnnTensorDescriptor_t m_d_bias_d;
  cudnnConvolutionDescriptor_t m_conv_fwd_d;
  cudnnConvolutionDescriptor_t m_conv_bwd_d;
  cudnnConvolutionDescriptor_t m_conv_bwd_filter_d;
  cudnnConvolutionFwdAlgo_t m_fwd_algo;
  cudnnConvolutionBwdDataAlgo_t m_bwd_data_algo;
  cudnnConvolutionBwdFilterAlgo_t m_bwd_filter_algo;
  size_t m_ws_size_fwd;
  size_t m_ws_size_bwd_data;
  size_t m_ws_size_bwd_filter;
  BoundaryAttributesV<size_t> m_ws_size_fwd_boundaries;
  IntVector m_halo_fwd_send;
  IntVector m_halo_bwd_send;
  IntVector m_halo_fwd_recv;
  IntVector m_halo_bwd_recv;

  HaloExchangeMethod m_halo_xch_method;
  using HaloExchange = tensor::HaloExchange<DataType,
                                            tensor::CUDAAllocator,
                                            Al::NCCLBackend>;
  using HaloExchangeMPI = tensor::HaloExchangeMPI<DataType,
                                                  tensor::CUDAAllocator,
                                                  Al::NCCLBackend>;
  using HaloExchangeAL = tensor::HaloExchangeAL<DataType,
                                                tensor::CUDAAllocator,
                                                Al::NCCLBackend>;
#ifdef DISTCONV_HAS_P2P
  using HaloExchangeP2P = tensor::HaloExchangeP2P<DataType,
                                                  tensor::CUDAAllocator,
                                                  Al::NCCLBackend>;
  using HaloExchangeHybrid = tensor::HaloExchangeHybrid<DataType,
                                                        tensor::CUDAAllocator,
                                                        Al::NCCLBackend>;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  using HaloExchangeNVSHMEM = tensor::HaloExchangeNVSHMEM<DataType,
                                                          tensor::CUDAAllocator,
                                                          Al::NCCLBackend>;
#ifdef DISTCONV_HAS_CUDA_GRAPH
  using HaloExchangeNVSHMEMGraph = tensor::HaloExchangeNVSHMEMGraph<DataType,
                                                                    tensor::CUDAAllocator,
                                                                    Al::NCCLBackend>;
#endif // DISTCONV_HAS_CUDA_GRAPH
  using HaloExchangeNVSHMEMDirect = tensor::HaloExchangeNVSHMEMDirect<DataType,
                                                                      tensor::CUDAAllocator,
                                                                      Al::NCCLBackend>;
  using HaloExchangeNVSHMEMFusedNotify = tensor::HaloExchangeNVSHMEMFusedNotify<
    DataType, tensor::CUDAAllocator, Al::NCCLBackend>;
#endif // DISTCONV_HAS_NVSHMEM
  std::unique_ptr<HaloExchange> m_halo_xch_input;
  std::unique_ptr<HaloExchange> m_halo_xch_d_output;

  bool m_overlap_halo_exchange_fwd;
  bool m_overlap_halo_exchange_bwd;
  cudnnTensorDescriptor_t m_input_interior_d;
  cudnnTensorDescriptor_t m_output_interior_d;
  bool m_interior_req;
  BoundaryAttributesV<bool> m_boundary_req = false;
  BoundaryAttributesV<cudnnTensorDescriptor_t> m_input_boundaries_d;
  BoundaryAttributesV<cudnnTensorDescriptor_t> m_output_boundaries_d;
  BoundaryAttributesV<cudnnConvolutionFwdAlgo_t> m_fwd_boundary_algos;
  index_t m_input_interior_offset = 0;
  index_t m_output_interior_offset = 0;
  BoundaryAttributesV<index_t> m_input_boundary_offsets = 0;
  BoundaryAttributesV<index_t> m_output_boundary_offsets = 0;
  BoundaryAttributesV<cudaStream_t> m_boundary_streams;
  BoundaryAttributesV<
    std::shared_ptr<Al::NCCLBackend::comm_type>> m_boundary_comms;

  bool m_enable_profiling;
  cudaEvent_t m_event_comp_start;
  cudaEvent_t m_event_comp_end;
  cudaEvent_t m_event_exchange_start;
  cudaEvent_t m_event_exchange_end;
  BoundaryAttributesV<cudaEvent_t> m_event_start_boundaries;
  BoundaryAttributesV<cudaEvent_t> m_event_end_boundaries;

  ChannelParallelismAlgorithm m_chanfilt_algo;
  // TODO: Maybe don't hardcode the allocator...
  tensor::Tensor<DataType, LocaleMPI, tensor::CUDAAllocator> m_input_gathered_t;
  cudnnTensorDescriptor_t m_input_gathered_d;
  tensor::Tensor<DataType, LocaleMPI, tensor::CUDAAllocator> m_output_all_filters_t;
  cudnnTensorDescriptor_t m_output_all_filters_d;
  tensor::Tensor<DataType, LocaleMPI, tensor::CUDAAllocator> m_d_output_gathered_t;
  cudnnTensorDescriptor_t m_d_output_gathered_d;
  tensor::Tensor<DataType, LocaleMPI, tensor::CUDAAllocator> m_d_input_all_channels_t;
  cudnnTensorDescriptor_t m_d_input_all_channels_d;
  index_t m_chanfilt_segments = 1;
  tensor::ChannelExchange<DataType> m_channel_exchange;

  void setup_profiling_events() {
    if (!m_enable_profiling) return;
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_event_comp_start));
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_event_comp_end));
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_event_exchange_start));
    DISTCONV_CHECK_CUDA(cudaEventCreate(&m_event_exchange_end));
    apply_to_spatial_sides(m_num_dims, [this](int i, Side side) {
                                         DISTCONV_CHECK_CUDA(cudaEventCreate(
                                             &m_event_start_boundaries(i, side)));
                                         DISTCONV_CHECK_CUDA(cudaEventCreate(
                                             &m_event_end_boundaries(i, side)));
                                       });
  }

  void record_start_comp() {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_comp_start,
                                          m_be.get_stream()));
    }
  }

  void record_end_comp() {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_comp_end,
                                          m_be.get_stream()));
    }
  }

  void record_start_exchange() {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_exchange_start,
                                          m_be.get_stream()));
    }
  }

  void record_end_exchange() {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_exchange_end,
                                          m_be.get_stream()));
    }
  }

  void record_start_boundary(int i, Side side) {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_start_boundaries(i, side),
                                          get_boundary_stream(i, side)));
    }
  }

  void record_end_boundary(int i, Side side) {
    if (m_enable_profiling) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(m_event_end_boundaries(i, side),
                                          get_boundary_stream(i, side)));
    }
  }

  void dump_profile_statistics(bool halo_exchange, bool has_boundary,
                               bool is_forward) {
    if (!m_enable_profiling) return;
    //DISTCONV_CHECK_CUDA(cudaStreamSynchronize(m_be.get_stream()));
    DISTCONV_CHECK_CUDA(cudaDeviceSynchronize());
    float main_elapsed;
    DISTCONV_CHECK_CUDA(cudaEventElapsedTime(
        &main_elapsed, m_event_comp_start, m_event_comp_end));
    std::stringstream ss;
    ss << "Convolution main: " << main_elapsed;
    if (halo_exchange) {
      DISTCONV_CHECK_CUDA(cudaEventElapsedTime(
          &main_elapsed, m_event_exchange_start, m_event_exchange_end));
      ss << ", halo exhange: " << main_elapsed;
    }
    if (has_boundary) {
      if ((is_forward && m_overlap_halo_exchange_fwd) ||
          (!is_forward && m_overlap_halo_exchange_bwd)) {
        apply_to_spatial_sides(
            m_num_dims,
            [&ss, this](int i, Side side) {
              if (!m_boundary_req(i, side)) return;
              float elapsed;
              DISTCONV_CHECK_CUDA(cudaEventElapsedTime(
                  &elapsed,
                  m_event_start_boundaries(i, side),
                  m_event_end_boundaries(i, side)));
              ss << ", boundary (" << i << ", " << side << "): " << elapsed;
            });
      }
    }
    util::MPIPrintStreamInfo() << ss.str();
  }

  void destroy_profiling_events() {
    if (!m_enable_profiling) return;
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_event_comp_start));
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_event_comp_end));
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_event_exchange_start));
    DISTCONV_CHECK_CUDA(cudaEventDestroy(m_event_exchange_end));
    apply_to_spatial_sides(
        m_num_dims,
        [this](int i, Side side) {
          DISTCONV_CHECK_CUDA(cudaEventDestroy(
              m_event_start_boundaries(i, side)));
          DISTCONV_CHECK_CUDA(cudaEventDestroy(
              m_event_end_boundaries(i, side)));
        });
  }

  template <typename Allocator>
  void setup_tensor_descriptors(const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &input,
                                const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &filter,
                                const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &output,
                                const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &d_input,
                                const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &d_filter,
                                const tensor::Tensor<DataType, LocaleMPI,
                                Allocator> &d_output,
                                const int_vector &strides,
                                const int_vector &dilations) {
    cudnn::setup_tensor_descriptor(m_input_d, input,
                                   m_halo_fwd_recv, m_halo_bwd_recv);
    cudnn::setup_tensor_descriptor(m_input_no_halo_d, input, false);
    if (!m_skip_bp_data) {
      cudnn::setup_tensor_descriptor(m_d_input_d, d_input,
                                     m_halo_fwd_recv, m_halo_bwd_recv);
    }
    setup_filter_descriptor(m_filter_d, filter);
    setup_filter_descriptor(m_d_filter_d, d_filter);
    cudnn::setup_tensor_descriptor(m_output_d, output, false);
    cudnn::setup_tensor_descriptor(m_d_output_d, d_output);
    cudnn::setup_tensor_descriptor(m_d_output_no_halo_d,
                                   d_output, false);

    // tensor descriptors for interior and boundary regions. only used
    // when overlapping is enabled
    if (m_overlap_halo_exchange_fwd) {
      setup_tensors_overlap(input, filter, output, strides, dilations);
    }

    setup_boundary_streams(input.get_split_index());
    util::MPIPrintStreamDebug() << "input: " << m_input_d;
    util::MPIPrintStreamDebug() << "input (no halo): "
                                << m_input_no_halo_d;
    if (m_overlap_halo_exchange_fwd) {
      util::MPIPrintStreamDebug() << "input interior: "
                                  << m_input_interior_d;
      apply_to_spatial_sides(
          m_num_dims,
          [this](int i, Side side) {
            if (m_boundary_req(i, side)) {
              util::MPIPrintStreamDebug() << "input boundary for dimension "
                                          << i << ", " << side << ": "
                                          << m_input_boundaries_d(i, side);
            }
          });
    }
    if (!m_skip_bp_data) {
      util::MPIPrintStreamDebug() << "d_input: " << m_d_input_d;
    }
    util::MPIPrintStreamDebug() << "filter: " << m_filter_d;
    util::MPIPrintStreamDebug() << "d_filter: " << m_d_filter_d;
    util::MPIPrintStreamDebug() << "output: " << m_output_d;
    if (m_overlap_halo_exchange_fwd) {
      util::MPIPrintStreamDebug() << "output interior: "
                                  << m_output_interior_d;
      apply_to_spatial_sides(
          m_num_dims,
          [this](int i , Side side) {
            if (m_boundary_req(i, side)) {
              util::MPIPrintStreamDebug() << "output boundary for dimension "
                                          << i << ", " << side << ": "
                                          << m_output_boundaries_d(i, side);
            }
          });
    }
    util::MPIPrintStreamDebug() << "d_output: " << m_d_output_d;
    util::MPIPrintStreamDebug() << "d_output (no halo): "
                                << m_d_output_no_halo_d;

    if (m_chanfilt_algo != ChannelParallelismAlgorithm::NONE) {
      setup_chanfilt_tensors(input, filter, output, d_input, d_filter, d_output);
      setup_chanfilt_comms(input, filter);
    }
  }

  template <typename Allocator>
  void setup_tensors_overlap(const tensor::Tensor<DataType, LocaleMPI,
                             Allocator> &input,
                             const tensor::Tensor<DataType, LocaleMPI,
                             Allocator> &filter,
                             const tensor::Tensor<DataType, LocaleMPI,
                             Allocator> &output,
                             const int_vector &strides,
                             const int_vector &dilations) {
    auto input_shape = input.get_local_shape();
    auto output_shape = output.get_local_shape();
    IndexVector input_interior_idx(m_num_dims, 0);
    IndexVector output_interior_idx(m_num_dims, 0);
    apply_to_spatial_sides(
        m_num_dims,
        [&](int dim, Side side) {
          auto filter_dim = internal::get_dilated_filter_size(
              static_cast<int>(filter.get_shape()[dim]), dilations[dim]);
          int st = strides[dim];
          int h = get_input_halo_recv(dim, side);
          // set default value
          m_boundary_req(dim, side) = false;
          if (h == 0) {
            // nothing to do
            util::MPIPrintStreamDebug() << "No boundary for " <<  dim << ", " << side;
            return;
          }
          util::MPIPrintStreamDebug() << dim << ", " << side << ": halo recv: " << h;
          int num_boundary_centers = util::ceil(h, st);
          int interior_offset = num_boundary_centers * st - h;
          int boundary_edge = ((h - 1) / st) * st;
          int input_boundary_dim = filter_dim + boundary_edge;
          if (side == LHS) {
            input_interior_idx[dim] = interior_offset;
            output_interior_idx[dim] = num_boundary_centers;
          }
          int output_boundary_dim = input_boundary_dim > 0 ? util::ceil(
              input_boundary_dim - (filter_dim - 1), st) : 0;
          assert_always(interior_offset >= 0);
          util::MPIPrintStreamDebug()
              << "Side: " << side << ", dim: " << dim << "; "
              << num_boundary_centers << ", " << h
              << ", " << interior_offset;
          input_shape[dim] -= interior_offset;
          output_shape[dim] -= num_boundary_centers;
          // boundary tensors
          if (input_boundary_dim == 0 || output_boundary_dim == 0) {
            m_boundary_req(dim, side) = false;
            util::MPIPrintStreamInfo()
                << "No boundary for dimension " << dim
                << ", side " << side;
            return;
          }
          m_boundary_req(dim, side) = true;
          cudnn::copy_tensor_descriptor(m_input_boundaries_d(dim, side),
                                        m_input_d);
          cudnn::set_tensor_dimension(m_input_boundaries_d(dim, side),
                                      dim, input_boundary_dim);
          cudnn::copy_tensor_descriptor(m_output_boundaries_d(dim, side),
                                        m_output_d);
          cudnn::set_tensor_dimension(m_output_boundaries_d(dim, side),
                                      dim, output_boundary_dim);
          setup_boundary_offsets(dim, side, input, output,
                                 input_boundary_dim, output_boundary_dim);
        });
    if (input_shape.is_empty() || output_shape.is_empty()) {
      m_interior_req = false;
    } else {
      m_interior_req = true;
      cudnn::setup_tensor_descriptor(m_input_interior_d, input,
                                     input_shape);
      m_input_interior_offset = input.get_local_offset(
          input_interior_idx, false);
      cudnn::setup_tensor_descriptor(m_output_interior_d,
                                     output, output_shape);
      m_output_interior_offset = output.get_local_offset(
          output_interior_idx, false);
    }
  }

  template <typename Allocator>
  void setup_chanfilt_tensors(const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &input,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &filter,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &output,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &d_input,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &d_filter,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &d_output) {
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          output.get_distribution().get_locale_shape());
        auto local_shape = output.get_local_shape();
        local_shape[-2] = output.get_shape()[-2];
        m_output_all_filters_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          output.get_shape(), output.get_locale(), dist,
          local_shape, tensor::Shape(output.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "output_all_filters tensor: " << m_output_all_filters_t;
        get_tmp_tensor_buffer(m_output_all_filters_t);
        cudnn::setup_tensor_descriptor(m_output_all_filters_d,
                                       m_output_all_filters_t,
                                       m_output_all_filters_t.get_local_shape());
        release_tmp_tensor_buffer(m_output_all_filters_t);
        util::MPIPrintStreamDebug() << "output_all_filters descriptor: " << m_output_all_filters_d;
      }
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          d_output.get_distribution().get_locale_shape());
        auto local_shape = d_output.get_local_shape();
        local_shape[-2] = d_output.get_shape()[-2];
        m_d_output_gathered_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          d_output.get_shape(), d_output.get_locale(), dist,
          local_shape, tensor::Shape(d_output.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "d_output_gathered tensor: " << m_d_output_gathered_t;
        get_tmp_tensor_buffer(m_d_output_gathered_t);
        cudnn::setup_tensor_descriptor(m_d_output_gathered_d,
                                       m_d_output_gathered_t,
                                       m_d_output_gathered_t.get_local_shape());
        release_tmp_tensor_buffer(m_d_output_gathered_t);
        util::MPIPrintStreamDebug() << "d_output_gathered descriptor: " << m_d_output_gathered_d;
      }
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          input.get_distribution().get_locale_shape());
        auto local_shape = input.get_local_shape();
        local_shape[-2] = input.get_shape()[-2];
        m_input_gathered_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          input.get_shape(), input.get_locale(), dist,
          local_shape, tensor::Shape(input.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "input_gathered tensor: " << m_input_gathered_t;
        get_tmp_tensor_buffer(m_input_gathered_t);
        cudnn::setup_tensor_descriptor(m_input_gathered_d,
                                       m_input_gathered_t,
                                       m_input_gathered_t.get_local_shape());
        release_tmp_tensor_buffer(m_input_gathered_t);
        util::MPIPrintStreamDebug() << "input_gathered descriptor: " << m_input_gathered_d;
      }
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          d_input.get_distribution().get_locale_shape());
        auto local_shape = d_input.get_local_shape();
        local_shape[-2] = d_input.get_shape()[-2];
        m_d_input_all_channels_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          d_input.get_shape(), d_input.get_locale(), dist,
          local_shape, tensor::Shape(d_input.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "d_input_all_channels tensor: " << m_d_input_all_channels_t;
        get_tmp_tensor_buffer(m_d_input_all_channels_t);
        cudnn::setup_tensor_descriptor(m_d_input_all_channels_d,
                                       m_d_input_all_channels_t,
                                       m_d_input_all_channels_t.get_local_shape());
        release_tmp_tensor_buffer(m_d_input_all_channels_t);
        util::MPIPrintStreamDebug() << "d_input_all_channels descriptor: " << m_d_input_all_channels_d;
      }
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      const auto filter_dim = filter.get_distribution().get_split_shape()[-1];
      const auto channel_dim = filter.get_distribution().get_split_shape()[-2];
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          input.get_distribution().get_locale_shape());
        auto local_shape = input.get_local_shape();
        local_shape[-2] = input.get_shape()[-2] / channel_dim;
        m_input_gathered_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          input.get_shape(), input.get_locale(), dist,
          local_shape, tensor::Shape(input.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "input_gathered tensor: " << m_input_gathered_t;
        get_tmp_tensor_buffer(m_input_gathered_t);
        cudnn::setup_tensor_descriptor(m_input_gathered_d,
                                       m_input_gathered_t,
                                       m_input_gathered_t.get_local_shape());
        release_tmp_tensor_buffer(m_input_gathered_t);
        util::MPIPrintStreamDebug() << "input_gathered descriptor: " << m_input_gathered_d;
      }
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          output.get_distribution().get_locale_shape());
        auto local_shape = output.get_local_shape();
        local_shape[-2] = output.get_shape()[-2] / filter_dim;
        m_output_all_filters_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          output.get_shape(), output.get_locale(), dist,
          local_shape, tensor::Shape(output.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "output_all_filters tensor: " << m_output_all_filters_t;
        get_tmp_tensor_buffer(m_output_all_filters_t);
        cudnn::setup_tensor_descriptor(m_output_all_filters_d,
                                       m_output_all_filters_t,
                                       m_output_all_filters_t.get_local_shape());
        release_tmp_tensor_buffer(m_output_all_filters_t);
        util::MPIPrintStreamDebug() << "output_all_filters descriptor: " << m_output_all_filters_d;
      }
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          d_output.get_distribution().get_locale_shape());
        auto local_shape = d_output.get_local_shape();
        local_shape[-2] = d_output.get_shape()[-2] / filter_dim;
        m_d_output_gathered_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          d_output.get_shape(), d_output.get_locale(), dist,
          local_shape, tensor::Shape(d_output.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "d_output_gathered tensor: " << m_d_output_gathered_t;
        get_tmp_tensor_buffer(m_d_output_gathered_t);
        cudnn::setup_tensor_descriptor(m_d_output_gathered_d,
                                       m_d_output_gathered_t,
                                       m_d_output_gathered_t.get_local_shape());
        release_tmp_tensor_buffer(m_d_output_gathered_t);
        util::MPIPrintStreamDebug() << "d_output_gathered descriptor: " << m_d_output_gathered_d;
      }
      {
        auto dist = tensor::Distribution::make_shared_distribution(
          d_input.get_distribution().get_locale_shape());
        auto local_shape = d_input.get_local_shape();
        local_shape[-2] = d_input.get_shape()[-2] / channel_dim;
        m_d_input_all_channels_t = tensor::Tensor<DataType, LocaleMPI, Allocator>(
          d_input.get_shape(), d_input.get_locale(), dist,
          local_shape, tensor::Shape(d_input.get_num_dims(), 0));
        util::MPIPrintStreamDebug() << "d_input_all_channels tensor: " << m_d_input_all_channels_t;
        get_tmp_tensor_buffer(m_d_input_all_channels_t);
        cudnn::setup_tensor_descriptor(m_d_input_all_channels_d,
                                       m_d_input_all_channels_t,
                                       m_d_input_all_channels_t.get_local_shape());
        release_tmp_tensor_buffer(m_d_input_all_channels_t);
        util::MPIPrintStreamDebug() << "d_input_all_channels descriptor: " << m_d_input_all_channels_d;
      }
    }
  }

  template <typename Allocator>
  void setup_chanfilt_comms(const tensor::Tensor<DataType, LocaleMPI,
                            Allocator> &input,
                            const tensor::Tensor<DataType, LocaleMPI,
                            Allocator> &filter) {
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::NONE) {
      return;
    }
    m_chanfilt_segments = input.get_distribution().get_split_shape()[-2];
    // All variants use the same communicator for allreduces.
    if (m_be.get_segmented_ar_comm(m_chanfilt_segments) == nullptr) {
      m_be.init_segmented_ar_comm(m_chanfilt_segments,
                                  input.get_sub_locale_except_dim(-2).get_comm());
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X ||
        m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      if (m_be.get_chanfilt_channel_comm(m_chanfilt_segments) == nullptr) {
        m_be.init_chanfilt_channel_comm(m_chanfilt_segments,
                                        input.get_sub_locale(-2).get_comm());
      }
      if (m_be.get_chanfilt_filter_comm(m_chanfilt_segments) == nullptr) {
        m_be.init_chanfilt_filter_comm(m_chanfilt_segments,
                                       input.get_sub_locale(-2).get_comm());
      }
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      if (m_be.get_chanfilt_channel_comm(m_chanfilt_segments) == nullptr) {
        m_be.init_chanfilt_channel_comm(m_chanfilt_segments,
                                        filter.get_sub_locale(-1).get_comm());
      }
      if (m_be.get_chanfilt_filter_comm(m_chanfilt_segments) == nullptr) {
        m_be.init_chanfilt_filter_comm(m_chanfilt_segments,
                                       filter.get_sub_locale(-2).get_comm());
      }
    }
  }

  void setup_algorithms(const std::string &fwd_algo,
                        const std::string &bwd_data_algo,
                        const std::string &bwd_filter_algo,
                        void *input,
                        void *filter,
                        void *output,
                        size_t ws_size=0) {
    setup_algorithms_fwd(fwd_algo, input, filter, output, ws_size);
    setup_algorithms_bwd(bwd_data_algo, bwd_filter_algo, input, filter, output, ws_size);
  }

  void setup_algorithms_fwd(const std::string &fwd_algo,
                            void *input,
                            void *filter,
                            void *output,
                            size_t ws_size=0) {
    // Note that m_bwd algo is set when deconv is used. Support for
    // deconv is partial.
    if (!m_overlap_halo_exchange_fwd) {
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
        ensure_tensor_descriptors_conform(
          m_input_d, m_output_all_filters_d, m_filter_d,
          "stationary-x setup algos forward");
        get_tmp_tensor_buffer(m_output_all_filters_t);
        m_fwd_algo = m_be.get_fwd_algorithm(
          fwd_algo, &m_input_d, input,
          &m_filter_d, filter,
          &m_conv_fwd_d, &m_output_all_filters_d,
          m_output_all_filters_t.get_buffer(),
          ws_size);
        release_tmp_tensor_buffer(m_output_all_filters_t);
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_d, m_filter_d,
          "stationary-y setup algos forward");
        get_tmp_tensor_buffer(m_input_gathered_t);
        m_fwd_algo = m_be.get_fwd_algorithm(
          fwd_algo, &m_input_gathered_d,
          m_input_gathered_t.get_buffer(),
          &m_filter_d, filter,
          &m_conv_fwd_d, &m_output_d, output,
          ws_size);
        release_tmp_tensor_buffer(m_input_gathered_t);
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_all_filters_d, m_filter_d,
          "stationary-w setup algos forward");
        get_tmp_tensor_buffer(m_input_gathered_t);
        get_tmp_tensor_buffer(m_output_all_filters_t);
        m_fwd_algo = m_be.get_fwd_algorithm(
          fwd_algo, &m_input_gathered_d,
          m_input_gathered_t.get_buffer(),
          &m_filter_d, filter,
          &m_conv_fwd_d, &m_output_all_filters_d,
          m_output_all_filters_t.get_buffer(),
          ws_size);
        release_tmp_tensor_buffer(m_input_gathered_t);
        release_tmp_tensor_buffer(m_output_all_filters_t);
      } else {
        if (!m_deconv) {
          ensure_tensor_descriptors_conform(
            m_input_d, m_output_d, m_filter_d,
            "setup algos forward");
          m_fwd_algo = m_be.get_fwd_algorithm(
              fwd_algo, &m_input_d, input, &m_filter_d, filter,
              &m_conv_fwd_d, &m_output_d, output, ws_size);
        } else {
          m_bwd_data_algo = m_be.get_bwd_data_algorithm(
              fwd_algo, &m_filter_d, filter, &m_input_d, input,
              &m_conv_fwd_d, &m_output_d, output, ws_size);
        }
      }
      util::MPIPrintStreamDebug()
          << "Convolution forward algorithm: "
          << (m_deconv ? util::get_name(m_bwd_data_algo) : util::get_name(m_fwd_algo));
    } else {
      if (m_interior_req) {
        m_fwd_algo = m_be.get_fwd_algorithm(
            fwd_algo, &m_input_interior_d, input,
            &m_filter_d, filter, &m_conv_fwd_d,
            &m_output_interior_d, output, ws_size);
        util::MPIPrintStreamDebug()
            << "Convolution forward interior algorithm: "
            << util::get_name(m_fwd_algo);
      }
      // TODO: Need to support this with chanfilt.
      apply_to_spatial_sides(
          m_num_dims,
          [&](int i, Side side) {
            if (m_boundary_req(i, side)) {
              // The workspace is reserved for the interior
              // sub-tensor. To be more robust, workspace sizes for
              // boundary regions should be set.
              m_fwd_boundary_algos(i, side) =
                  m_be.get_fwd_algorithm(fwd_algo,
                                         &m_input_boundaries_d(i, side), input,
                                         &m_filter_d, filter,
                                         &m_conv_fwd_d,
                                         &m_output_boundaries_d(i, side), output,
                                         0);
              util::MPIPrintStreamDebug()
                  << "Convolution forward boundary algorithm for (" << i << ", "
                  << side << "): "
                  << util::get_name(m_fwd_boundary_algos(i, side));
            }
          });
    }
  }

  void setup_algorithms_bwd(const std::string &bwd_data_algo,
                            const std::string &bwd_filter_algo,
                            void *input,
                            void *filter,
                            void *output,
                            size_t ws_size=0) {
    // Similarly to setup_algorithms_fwd, m_fwd_algo is set when
    // deconv is used.
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
      get_tmp_tensor_buffer(m_d_output_gathered_t);
      if (!m_skip_bp_data) {
        ensure_tensor_descriptors_conform(
          m_d_input_d, m_d_output_gathered_d, m_filter_d,
          "stationary-x setup algos backward-data");
        m_bwd_data_algo = m_be.get_bwd_data_algorithm(
            bwd_data_algo, &m_filter_d, filter, &m_d_output_gathered_d,
            m_d_output_gathered_t.get_buffer(),
            &m_conv_bwd_d, &m_d_input_d, input, ws_size);
      }
      ensure_tensor_descriptors_conform(
          m_input_d, m_d_output_gathered_d, m_filter_d,
          "stationary-x setup algos backward-filter");
      m_bwd_filter_algo = m_be.get_bwd_filter_algorithm(
          bwd_filter_algo, &m_input_d, input, &m_d_output_gathered_d,
          m_d_output_gathered_t.get_buffer(),
          &m_conv_bwd_filter_d, &m_d_filter_d, filter, ws_size);
      release_tmp_tensor_buffer(m_d_output_gathered_t);
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      get_tmp_tensor_buffer(m_input_gathered_t);
      get_tmp_tensor_buffer(m_d_input_all_channels_t);
      if (!m_skip_bp_data) {
        ensure_tensor_descriptors_conform(
          m_d_input_all_channels_d, m_d_output_d, m_filter_d,
          "stationary-y setup algos backward-data");
        m_bwd_data_algo = m_be.get_bwd_data_algorithm(
            bwd_data_algo, &m_filter_d, filter, &m_d_output_d, output,
            &m_conv_bwd_d, &m_d_input_all_channels_d,
            m_d_input_all_channels_t.get_buffer(), ws_size);
      }
      ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_d_output_d, m_filter_d,
          "stationary-y setup algos backward-filter");
      m_bwd_filter_algo = m_be.get_bwd_filter_algorithm(
          bwd_filter_algo, &m_input_gathered_d, m_input_gathered_t.get_buffer(),
          &m_d_output_d, output,
          &m_conv_bwd_filter_d, &m_d_filter_d, filter, ws_size);
      release_tmp_tensor_buffer(m_input_gathered_t);
      release_tmp_tensor_buffer(m_d_input_all_channels_t);
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      get_tmp_tensor_buffer(m_input_gathered_t);
      get_tmp_tensor_buffer(m_d_output_gathered_t);
      get_tmp_tensor_buffer(m_d_input_all_channels_t);
      if (!m_skip_bp_data) {
        ensure_tensor_descriptors_conform(
          m_d_input_all_channels_d, m_d_output_gathered_d, m_filter_d,
          "stationary-w setup algos backward-data");
        m_bwd_data_algo = m_be.get_bwd_data_algorithm(
          bwd_data_algo, &m_filter_d, filter, &m_d_output_gathered_d,
          m_d_output_gathered_t.get_buffer(),
          &m_conv_bwd_d, &m_d_input_all_channels_d,
          m_d_input_all_channels_t.get_buffer(), ws_size);
      }
      ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_d_output_gathered_d, m_filter_d,
          "stationary-w setup algos backward-filter");
      m_bwd_filter_algo = m_be.get_bwd_filter_algorithm(
        bwd_filter_algo, &m_input_gathered_d, m_input_gathered_t.get_buffer(),
        &m_d_output_gathered_d, m_d_output_gathered_t.get_buffer(),
        &m_conv_bwd_filter_d, &m_d_filter_d, filter, ws_size);
      release_tmp_tensor_buffer(m_input_gathered_t);
      release_tmp_tensor_buffer(m_d_output_gathered_t);
      release_tmp_tensor_buffer(m_d_input_all_channels_t);
    } else {
      if (!m_skip_bp_data) {
        if (!m_deconv) {
          ensure_tensor_descriptors_conform(
            m_d_input_d, m_d_output_d, m_filter_d,
            "setup algos backward-data");
          m_bwd_data_algo = m_be.get_bwd_data_algorithm(
              bwd_data_algo, &m_filter_d, filter, &m_d_output_d, output,
              &m_conv_bwd_d, &m_d_input_d, input, ws_size);
        } else {
          m_fwd_algo = m_be.get_fwd_algorithm(
              bwd_data_algo, &m_d_output_d, output, &m_filter_d, filter,
              &m_conv_bwd_d, &m_d_input_d, input, ws_size);
        }
      }
      if (!m_deconv) {
        ensure_tensor_descriptors_conform(
          m_input_d, m_d_output_d, m_filter_d,
          "setup algos backward-filter");
        m_bwd_filter_algo = m_be.get_bwd_filter_algorithm(
            bwd_filter_algo, &m_input_d, input, &m_d_output_d, output,
            &m_conv_bwd_filter_d, &m_d_filter_d, filter, ws_size);
      } else {
        m_bwd_filter_algo = m_be.get_bwd_filter_algorithm(
            bwd_filter_algo, &m_d_output_d, output, &m_input_d, input,
            &m_conv_bwd_filter_d, &m_d_filter_d, filter, ws_size);
      }
    }
    if (!m_skip_bp_data) {
      util::MPIPrintStreamDebug()
          << "Convolution backward data algorithm: "
          << (m_deconv ? util::get_name(m_fwd_algo) : util::get_name(m_bwd_data_algo));
    }
    util::MPIPrintStreamDebug()
        << "Convolution backward filter algorithm: "
        << util::get_name(m_bwd_filter_algo);
  }

  void setup_workspace_sizes() {
    setup_workspace_size_fwd();
    setup_workspace_size_fwd_boundaries();
    if (!m_skip_bp_data) setup_workspace_size_bwd_data();
    setup_workspace_size_bwd_filter();
  }

  void setup_workspace_size_fwd() {
    size_t s;
    util::MPIPrintStreamDebug()
        << "setup_workspace_size_fwd; "
        << "input: " << m_input_d
        << ", filter: " << m_filter_d
        << ", conv desc: " << m_conv_fwd_d
        << ", output: " << m_output_d;
    if (!m_overlap_halo_exchange_fwd) {
      if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
        ensure_tensor_descriptors_conform(
          m_input_d, m_output_all_filters_d, m_filter_d,
          "stationary-x workspace forward");
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                               m_be.get_handle(), m_input_d, m_filter_d, m_conv_fwd_d,
                               m_output_all_filters_d, m_fwd_algo, &s));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_d, m_filter_d,
          "stationary-y workspace forward");
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                               m_be.get_handle(), m_input_gathered_d, m_filter_d, m_conv_fwd_d,
                               m_output_d, m_fwd_algo, &s));
      } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
        ensure_tensor_descriptors_conform(
          m_input_gathered_d, m_output_all_filters_d, m_filter_d,
          "stationary-w workspace forward");
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                               m_be.get_handle(), m_input_gathered_d, m_filter_d, m_conv_fwd_d,
                               m_output_all_filters_d, m_fwd_algo, &s));
      } else {
        ensure_tensor_descriptors_conform(
          m_input_d, m_output_d, m_filter_d,
          "workspace forward");
        s = get_workspace_size_fwd(m_input_d, m_filter_d, m_output_d);
      }
    } else {
      if (m_interior_req) {
        // TODO: Handle with chanfilt.
        DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
            m_be.get_handle(), m_input_interior_d, m_filter_d,
            m_conv_fwd_d, m_output_interior_d, m_fwd_algo, &s));
      } else {
        s = 0;
      }
    }
    m_ws_size_fwd = s;
  }

  size_t get_workspace_size_fwd(cudnnTensorDescriptor_t input,
                                cudnnFilterDescriptor_t filter,
                                cudnnTensorDescriptor_t output) {
    size_t s;
    if (!m_deconv) {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
          m_be.get_handle(), input, filter, m_conv_fwd_d, output, m_fwd_algo, &s));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
          m_be.get_handle(), filter, input, m_conv_fwd_d, output, m_bwd_data_algo, &s));
    }
    return s;
  }

  void setup_workspace_size_fwd_boundaries() {
    // TODO: Handle with chanfilt.
    if (!m_overlap_halo_exchange_fwd) return;
    apply_to_spatial_sides(
        m_num_dims,
        [this](int i, Side side) {
          if (m_boundary_req(i, side)) {
            size_t s;
            DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                m_be.get_handle(), m_input_boundaries_d(i, side),
                m_filter_d, m_conv_fwd_d,
                m_output_boundaries_d(i, side),
                m_fwd_boundary_algos(i, side), &s));
            m_ws_size_fwd_boundaries(i, side) = s;
          }});
  }

  void setup_workspace_size_bwd_data() {
    if (m_skip_bp_data) return;
    size_t s;
    cudnnStatus_t err = CUDNN_STATUS_SUCCESS;
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
      ensure_tensor_descriptors_conform(
        m_d_input_d, m_d_output_gathered_d, m_filter_d,
        "stationary-x workspace backward-data");
      err = cudnnGetConvolutionBackwardDataWorkspaceSize(
        m_be.get_handle(), m_filter_d, m_d_output_gathered_d,
        m_conv_bwd_d,
        m_d_input_d, m_bwd_data_algo, &s);
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      ensure_tensor_descriptors_conform(
        m_d_input_all_channels_d, m_d_output_d, m_filter_d,
        "stationary-y workspace backward-data");
      err = cudnnGetConvolutionBackwardDataWorkspaceSize(
        m_be.get_handle(), m_filter_d, m_d_output_d,
        m_conv_bwd_d,
        m_d_input_all_channels_d, m_bwd_data_algo, &s);
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      ensure_tensor_descriptors_conform(
        m_d_input_all_channels_d, m_d_output_gathered_d, m_filter_d,
        "stationary-w workspace backward-data");
      err = cudnnGetConvolutionBackwardDataWorkspaceSize(
        m_be.get_handle(), m_filter_d, m_d_output_gathered_d,
        m_conv_bwd_d,
        m_d_input_all_channels_d, m_bwd_data_algo, &s);
    } else {
      ensure_tensor_descriptors_conform(
        m_d_input_d, m_d_output_d, m_filter_d,
        "workspace backward-data");
      s = get_workspace_size_bwd_data(m_filter_d, m_d_output_d, m_d_input_d);
    }
    if (err != CUDNN_STATUS_SUCCESS) {
      util::MPIPrintStreamError()
          << "Error at setup_workspace_size_bwd_data; "
          << "filter: " << util::tostring(m_filter_d)
          << ", d_output: " << util::tostring(m_d_output_d)
          << ", conv bwd: " << util::tostring(m_conv_bwd_d)
          << ", d_input: " << util::tostring(m_d_input_d);
      DISTCONV_CHECK_CUDNN(err);
    }
    m_ws_size_bwd_data = s;
  }

  size_t get_workspace_size_bwd_data(cudnnFilterDescriptor_t filter,
                                     cudnnTensorDescriptor_t d_output,
                                     cudnnTensorDescriptor_t d_input) {
    size_t s;
    if (!m_deconv) {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
          m_be.get_handle(), filter, d_output,
          m_conv_bwd_d, d_input, m_bwd_data_algo, &s));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(
          m_be.get_handle(), d_output, filter,
          m_conv_bwd_d, d_input, m_fwd_algo, &s));
    }
    return s;
  }

  void setup_workspace_size_bwd_filter() {
    size_t s;
    util::MPIPrintStreamDebug()
        << "setup_workspace_size_bwd_filter; "
        << "input_no_halo: " << m_input_no_halo_d
        << ", d_output: " << m_d_output_d
        << ", conv desc: " << m_conv_bwd_filter_d
        << ", d_filter: " << m_d_filter_d;
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::X) {
      ensure_tensor_descriptors_conform(
        m_input_d, m_d_output_gathered_d, m_d_filter_d,
        "stationary-x workspace backward-filter");
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                             m_be.get_handle(), m_input_d, m_d_output_gathered_d,
                             m_conv_bwd_filter_d, m_d_filter_d, m_bwd_filter_algo, &s));
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::Y) {
      ensure_tensor_descriptors_conform(
        m_input_gathered_d, m_d_output_d, m_d_filter_d,
        "stationary-y workspace backward-filter");
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                             m_be.get_handle(), m_input_gathered_d, m_d_output_d,
                             m_conv_bwd_filter_d, m_d_filter_d, m_bwd_filter_algo, &s));
    } else if (m_chanfilt_algo == ChannelParallelismAlgorithm::W) {
      ensure_tensor_descriptors_conform(
        m_input_gathered_d, m_d_output_gathered_d, m_d_filter_d,
        "stationary-w workspace backward-filter");
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
                             m_be.get_handle(), m_input_gathered_d, m_d_output_gathered_d,
                             m_conv_bwd_filter_d, m_d_filter_d, m_bwd_filter_algo, &s));
    } else {
      ensure_tensor_descriptors_conform(
        m_input_d, m_d_output_d, m_d_filter_d,
        "workspace backward-filter");
      s = get_workspace_size_bwd_filter(m_input_d, m_d_output_d, m_d_filter_d);
    }
    m_ws_size_bwd_filter = s;
  }

  size_t get_workspace_size_bwd_filter(cudnnTensorDescriptor_t input,
                                       cudnnTensorDescriptor_t d_output,
                                       cudnnFilterDescriptor_t d_filter) {
    size_t s = 0;
    if (!m_deconv) {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          m_be.get_handle(), input, d_output,
          m_conv_bwd_filter_d, d_filter, m_bwd_filter_algo, &s));
    } else {
      DISTCONV_CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
          m_be.get_handle(), d_output, input,
          m_conv_bwd_filter_d, d_filter, m_bwd_filter_algo, &s));
    }
    return s;
  }

  template <typename TensorType>
  void setup_filter_descriptor(cudnnFilterDescriptor_t &desc, const TensorType &tensor) {
    cudnnDataType_t dt = util::get_cudnn_type<typename TensorType::data_type>();
    const int_vector shape = tensor.get_local_real_shape().template get_vector<int>();
    assert_eq((unsigned int) m_num_dims, shape.size());
    DISTCONV_CHECK_CUDNN(cudnnSetFilterNdDescriptor(
        desc, dt, CUDNN_TENSOR_NCHW, m_num_dims,
        util::reverse(shape).data()));
  }

  void setup_convolution_descriptor(const IntVector &overlap,
                                    const tensor::Shape &filter_shape,
                                    const int_vector &pads,
                                    const int_vector &strides,
                                    const int_vector &dilations,
                                    int num_groups,
                                    cudnnConvolutionDescriptor_t &desc,
                                    cudnnConvolutionDescriptor_t &desc_bp_data,
                                    cudnnConvolutionDescriptor_t &desc_bp_filter) {

    cudnnConvolutionMode_t mode = CUDNN_CROSS_CORRELATION;
    cudnnDataType_t dt = util::get_cudnn_type<DataType>();

    for (int i = 0; i < m_num_spatial_dims; ++i) {
      auto df = internal::get_dilated_filter_size((int)filter_shape[i],
                                                  dilations[i]);
      if (!(pads[i] * 2 + 1 == df || pads[i] == 0)) {
        util::MPIPrintStreamError()
            << "Padding size must be zero or must match the filter size: "
            << "padding_w: " << pads[i]
            << ", filter shape: " << filter_shape;
        std::abort();
      }
    }

    auto stencil_dims = filter_shape;
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      auto window_dim = internal::get_dilated_filter_size<int>(
          stencil_dims[i], dilations[i]);
      if (window_dim % 2) {
        stencil_dims[i] = (window_dim - 1) / 2;
      } else {
        assert_eq(window_dim, strides[i]);
        stencil_dims[i] = 0;
      }
    }

    // Case without padding should work if stride is 1. Stride > 1 is
    // not considered yet. Should be fine if it's a 1x1 filter.
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      assert_always(pads[i] != 0 || strides[i] == 1 || stencil_dims[i] == 0);
    }

    int_vector pads_fp, pads_bp;
    pads_fp = pads;

    // TODO: stride when padding size is zero
    // NOTE: padding == 0 likely not working
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      if (pads[i] == 0) {
        pads_bp.push_back(stencil_dims[i]);
      } else {
        pads_bp.push_back(stencil_dims[i] * strides[i]);
      }
    }

    for (int i = 0; i < m_num_spatial_dims; ++i) {
      // when the input tensor is extended with halo, no padding is
      // necessary
      if (overlap[i] > 0) {
        pads_fp[i] = 0;
      } else {
        // if overlap is zero, don't manipulate padding
        pads_fp[i] = pads[i];
        pads_bp[i] = pads[i];
      }
    }

    const auto r_pads_fp   = util::reverse(pads_fp);
    const auto r_pads_bp   = util::reverse(pads_bp);
    const auto r_strides   = util::reverse(strides);
    const auto r_dilations = util::reverse(dilations);

    DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
        desc, m_num_spatial_dims,
        r_pads_fp.data(), r_strides.data(), r_dilations.data(),
        mode, dt));
    DISTCONV_CHECK_CUDNN(cudnnSetConvolutionGroupCount(desc, num_groups));

    if (!m_skip_bp_data) {
      DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
          desc_bp_data, m_num_spatial_dims,
          r_pads_bp.data(), r_strides.data(), r_dilations.data(),
          mode, dt));
      DISTCONV_CHECK_CUDNN(cudnnSetConvolutionGroupCount(
          desc_bp_data, num_groups));
    }

    // Note: Backward filter uses the same setting as backward data. This
    // requiers the halo region of the d_output to be zero-cleared. It
    // would be possible to avoid it by ignoring the halo region and
    // with appropriate length of padding. However, the required
    // lenght of padding can differ between the two ends of dimension,
    // and cuDNN only supports the same padding size, it won't
    // work. Instead, take the whole area as input including halo, and
    // zero-clear it before.
    DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
        desc_bp_filter, m_num_spatial_dims,
        r_pads_bp.data(), r_strides.data(), r_dilations.data(),
        mode, dt));
    DISTCONV_CHECK_CUDNN(cudnnSetConvolutionGroupCount(desc_bp_filter, num_groups));
  }

  template <typename Allocator>
  void setup_halo_xch(tensor::Tensor<DataType, LocaleMPI,
                      Allocator> &input,
                      tensor::Tensor<DataType, LocaleMPI,
                      Allocator> &d_output) {
    util::MPIRootPrintStreamDebug() << "Using " << m_halo_xch_method
                                    << " in halo exchange";
    switch (m_halo_xch_method) {
      case HaloExchangeMethod::MPI:
        m_halo_xch_input.reset(new HaloExchangeMPI(input));
        m_halo_xch_d_output.reset(new HaloExchangeMPI(d_output));
        break;
      case HaloExchangeMethod::AL:
        m_halo_xch_input.reset(new HaloExchangeAL(input));
        m_halo_xch_d_output.reset(new HaloExchangeAL(d_output));
        break;
#ifdef DISTCONV_HAS_P2P
      case HaloExchangeMethod::P2P:
        m_halo_xch_input.reset(new HaloExchangeP2P(input, m_be.get_p2p()));
        m_halo_xch_d_output.reset(new HaloExchangeP2P(d_output, m_be.get_p2p()));
        break;
      case HaloExchangeMethod::HYBRID:
        m_halo_xch_input.reset(new HaloExchangeHybrid(input, m_be.get_p2p()));
        m_halo_xch_d_output.reset(new HaloExchangeHybrid(d_output, m_be.get_p2p()));
        break;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
      case HaloExchangeMethod::NVSHMEM:
        m_halo_xch_input.reset(new HaloExchangeNVSHMEM(input));
        m_halo_xch_d_output.reset(new HaloExchangeNVSHMEM(d_output));
        break;
#ifdef DISTCONV_HAS_CUDA_GRAPH
      case HaloExchangeMethod::NVSHMEM_GRAPH:
        m_halo_xch_input.reset(new HaloExchangeNVSHMEMGraph(input));
        m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMGraph(d_output));
        break;
#endif // DISTCONV_HAS_CUDA_GRAPH
      case HaloExchangeMethod::NVSHMEM_DIRECT:
        m_halo_xch_input.reset(new HaloExchangeNVSHMEMDirect(input));
        m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMDirect(d_output));
        break;
      case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
        m_halo_xch_input.reset(new HaloExchangeNVSHMEMFusedNotify(input));
        m_halo_xch_d_output.reset(new HaloExchangeNVSHMEMFusedNotify(d_output));
        break;
#endif // DISTCONV_HAS_NVSHMEM
      default:
        util::MPIPrintStreamError()
            << "Invalid halo exchange method: " << m_halo_xch_method;
        std::abort();
    }
  }

  template <typename Allocator>
  void exchange_halo(tensor::Tensor<DataType, LocaleMPI,
                     Allocator> &tensor,
                     std::unique_ptr<HaloExchange> &xch,
                     bool async,
                     bool skip_unpack=false) {
    record_start_exchange();
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePushA("conv/forward/exchange_halo");
    }
    assert_always(xch != nullptr);
    xch->exchange(m_boundary_comms, m_be.get_stream(),
                  false, !async, false, skip_unpack);
    if (m_be.is_nvtx_enabled()) {
      nvtxRangePop();
    }
    record_end_exchange();
  }

  template <typename Allocator>
  void unpack_halo(tensor::Tensor<DataType, LocaleMPI,
                   Allocator> &tensor,
                   std::unique_ptr<HaloExchange> &xch) {
    xch->unpack(m_boundary_streams, m_be.get_stream(),
                true, false);
  }


  void wait_boundaries(cudaStream_t s) {
    apply_to_spatial_sides(m_num_dims,
                           [&](int i, Side side) {
                             if (m_boundary_req(i, side)) {
                               util::wait_stream(m_boundary_streams(i, side), s);
                             }
                           });
  }

  int get_input_halo_recv(int dim, Side side) {
    return side == LHS ? m_halo_bwd_recv[dim] :
        m_halo_fwd_recv[dim];
  }

  template <typename Allocator>
  void setup_boundary_offsets(int dim, Side side,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &input,
                              const tensor::Tensor<DataType, LocaleMPI,
                              Allocator> &output,
                              int input_boundary_dim,
                              int output_boundary_dim) {
    if (side == LHS) {
      m_input_boundary_offsets(dim, side) =
          input.get_local_offset() -
          input.get_local_offset(m_halo_bwd_recv, true);
      m_output_boundary_offsets(dim, side) = output.get_local_offset();
    } else {
      IndexVector input_boundary_idx = input.get_overlap() - m_halo_bwd_recv;
      input_boundary_idx[dim] = input.get_local_shape()[dim]
          + get_input_halo_recv(dim, side)
          - input_boundary_dim + input.get_overlap()[dim];
      m_input_boundary_offsets(dim, side) =
          input.get_local_offset(input_boundary_idx, true);
      IndexVector output_boundary_idx(m_num_dims, 0);
      output_boundary_idx[dim] =
          output.get_local_shape()[dim] - output_boundary_dim;
      m_output_boundary_offsets(dim, side) =
          output.get_local_offset(output_boundary_idx, false);
    }
  }

  void setup_boundary_streams(const IndexVector &split_idx) {
    apply_to_spatial_sides(
        m_num_dims,
        [this](int i, Side side) {
          int idx = get_boundary_stream_index(i, side);
          m_boundary_streams(i, side) =
              m_be.get_internal_stream_pr(idx);
          m_boundary_comms(i, side) =
              m_be.get_internal_al_mpi_cuda_comm(idx);
        });
    for (int i = 0; i < m_num_spatial_dims; ++i) {
      if (split_idx[i] % 2) {
        std::swap(m_boundary_streams(i, LHS), m_boundary_streams(i, RHS));
        std::swap(m_boundary_comms(i, LHS), m_boundary_comms(i, RHS));
      }
    }
  }

  int get_boundary_stream_index(int dim, Side side) {
    return dim * 2 + (side == LHS ? 0 : 1);
  }

  cudaStream_t get_boundary_stream(int i, Side side) {
    return m_boundary_streams(i, side);
  }

  template <typename Allocator>
  void allreduce_gradients(
      tensor::Tensor<DataType, LocaleMPI, Allocator> &gradients) {
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::NONE) {
      Al::Allreduce<Al::NCCLBackend, DataType>(
        gradients.get_base_ptr(),
        gradients.get_size(),
        Al::ReductionOperator::sum,
        m_be.get_al_nccl_comm());
    } else {
      Al::Allreduce<Al::NCCLBackend, DataType>(
        gradients.get_base_ptr(),
        gradients.get_local_size(),
        Al::ReductionOperator::sum,
        *m_be.get_segmented_ar_comm(m_chanfilt_segments));
    }
  }

  template <typename Allocator>
  void select_chanfilt_algorithm(
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &input,
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &filter,
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &output) {
    if (input.get_distribution().get_split_shape()[-2] == 1) {
      // Not partitioned along channels: no parallelism.
      m_chanfilt_algo = ChannelParallelismAlgorithm::NONE;
      return;
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::NONE) {
      std::cerr << "Channel/filter parallelism algorithm is NONE, but channels are partitioned\n";
      std::abort();
    }
    if (m_chanfilt_algo == ChannelParallelismAlgorithm::AUTO) {
      // Force this for now.
      m_chanfilt_algo = ChannelParallelismAlgorithm::X;
    }
  }

  /** Assemble the channel/filter dimension of src into dst. */
  template <typename Allocator>
  void allgather_chanfilt(
    tensor::Tensor<DataType, LocaleMPI, Allocator> &src,
    tensor::Tensor<DataType, LocaleMPI, Allocator> &dst,
    bool channel) {
    m_channel_exchange.allgather(
      src, dst,
      channel ?
      *m_be.get_chanfilt_channel_comm(m_chanfilt_segments) :
      *m_be.get_chanfilt_filter_comm(m_chanfilt_segments),
      m_be.get_stream());
  }

  /** Reduce-scatter the channel/filter dimension of src into dst. */
  template <typename Allocator>
  void reduce_scatter_chanfilt(
    tensor::Tensor<DataType, LocaleMPI, Allocator> &src,
    tensor::Tensor<DataType, LocaleMPI, Allocator> &dst,
    bool channel) {
    m_channel_exchange.reduce_scatter(
      src, dst,
      channel ?
      *m_be.get_chanfilt_channel_comm(m_chanfilt_segments) :
      *m_be.get_chanfilt_filter_comm(m_chanfilt_segments),
      m_be.get_stream());
  }

  /** Get a temporary buffer and set t to view it. */
  template <typename Allocator>
  void get_tmp_tensor_buffer(
    tensor::Tensor<DataType, LocaleMPI, Allocator> &t) {
    DataType *buf = (DataType *) internal::RuntimeCUDA::get_device_memory_pool().get(
      t.get_local_size()*sizeof(DataType), m_be.get_stream());
    tensor::View(t, buf);
  }

  /** Release a temporary buffer allocated for t. */
  template <typename Allocator>
  void release_tmp_tensor_buffer(
    tensor::Tensor<DataType, LocaleMPI, Allocator> &t) {
    internal::RuntimeCUDA::get_device_memory_pool().release(t.get_buffer());
    tensor::View(t, (DataType *) nullptr);
  }

  /**
   * Ensure tensor dimensions match.
   *
   * channel_tensor is either x or dL/dx.
   * filter_tensor is either y or dL/dy.
   * weights_tensor is either w or dL/dw.
   */
  template <typename Allocator>
  void ensure_tensors_conform(
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &channel_tensor,
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &filter_tensor,
    const tensor::Tensor<DataType, LocaleMPI, Allocator> &weights_tensor,
    const std::string &context) {
    auto c_shape = channel_tensor.get_local_shape();
    auto f_shape = filter_tensor.get_local_shape();
    auto w_shape = weights_tensor.get_local_shape();
    if (c_shape[-2] != w_shape[-2] || f_shape[-2] != w_shape[-1]) {
      util::MPIPrintStreamError()
        << context << ": tensors do not match:\n"
        << " channel_tensor: " << channel_tensor << "\n"
        << " filter_tensor: " << filter_tensor << "\n"
        << " weights_tensor: " << weights_tensor;
      std::abort();
    }
  }

  /** Like ensure_tensors_conform but for descriptors. */
  void ensure_tensor_descriptors_conform(
    const cudnnTensorDescriptor_t &channel_d,
    const cudnnTensorDescriptor_t &filter_d,
    const cudnnFilterDescriptor_t &weights_d,
    const std::string &context) {
    if (m_num_dims == 4) {
      if (cudnn::get_tensor_dimension(channel_d, -2) !=
          cudnn::get_filter_descriptor_dimension<4>(weights_d, -2) ||
          cudnn::get_tensor_dimension(filter_d, -2) !=
          cudnn::get_filter_descriptor_dimension<4>(weights_d, -1)) {
        util::MPIPrintStreamError()
          << context << ": descriptors do not match:\n"
          << " channel: " << channel_d << "\n"
          << " filter: " << filter_d << "\n"
          << " weights: " << weights_d;
        std::abort();
      }
    } else if (m_num_dims == 5) {
      if (cudnn::get_tensor_dimension(channel_d, -2) !=
          cudnn::get_filter_descriptor_dimension<5>(weights_d, -2) ||
          cudnn::get_tensor_dimension(filter_d, -2) !=
          cudnn::get_filter_descriptor_dimension<5>(weights_d, -1)) {
        util::MPIPrintStreamError()
          << context << ": descriptors do not match:\n"
          << " channel: " << channel_d << "\n"
          << " filter: " << filter_d << "\n"
          << " weights: " << weights_d;
        std::abort();
      }
    } else {
      util::MPIPrintStreamError()
        << "Unsupported num_dims " << m_num_dims;
      std::abort();
    }
  }

};

} // namespace distconv
