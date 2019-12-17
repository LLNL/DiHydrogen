#pragma once

#include "distconv/base.hpp"
#include "distconv/layers.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_cudnn.hpp"
#include "distconv/util/util_cuda.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/tensor_cuda.hpp"
#include "distconv/tensor/tensor_mpi_cuda.hpp"

#ifdef DISTCONV_HAS_P2P
#include "p2p/p2p.hpp"
#endif // DISTCONV_HAS_P2P
#include <Al.hpp>

#include <string>
#include <type_traits>
#include <memory>

#include <cudnn.h>
#include <nvToolsExt.h>
#include <cuda_profiler_api.h>

namespace distconv {
namespace cudnn {

constexpr int nb_dims_requested = 100;

template <typename Tensor, typename ShapeType>
inline void setup_tensor_descriptor(
    cudnnTensorDescriptor_t &desc,
    const Tensor &tensor, const ShapeType &shape) {
  cudnnDataType_t dt = util::get_cudnn_type<typename Tensor::data_type>();
  assert_eq(tensor.get_num_dims(), shape.num_dims());

  if (shape.get_size() == 0) return;

  // set descriptor for input tensor
  // The size should include halo regions. Convolution will not be
  // done for the halo regions by disabling padding
  IndexVector strides = tensor::get_strides(tensor.get_local_shape(),
                                            tensor.get_halo_width(),
                                            tensor.get_pitch());

  util::MPIPrintStreamDebug()
      << "setup_tensor_descriptor. "
      << "tensor: " << tensor
      << ", shape: " << util::join_array(shape, ", ")
      << ", strides: " << util::join_array(strides, ", ") << "\n";

  DISTCONV_CHECK_CUDNN(cudnnSetTensorNdDescriptor(
      desc, dt, shape.num_dims(),
      util::reverse(IntVector(shape)).data(),
      util::reverse(strides).get_vector<int>().data()));
}

template <typename Tensor>
inline void setup_tensor_descriptor(
    cudnnTensorDescriptor_t &desc,
    const Tensor &tensor,
    const IntVector &halo_fwd,
    const IntVector &halo_bwd) {
  auto shape = tensor.get_local_shape();
  shape = shape + tensor::Shape(halo_fwd) + tensor::Shape(halo_bwd);
  return setup_tensor_descriptor(desc, tensor, shape);
}

template <typename Tensor>
inline void setup_tensor_descriptor(
    cudnnTensorDescriptor_t &desc,
    const Tensor &tensor,
    const std::vector<bool> &include_halo_fwd,
    const std::vector<bool> &include_halo_bwd) {
  const int nd = tensor.get_num_dims();
  auto overlap = tensor.get_overlap();
  IntVector halo_fwd(nd, 0), halo_bwd(nd, 0);
  for (int i = 0; i < nd; ++i) {
    if (include_halo_bwd[i]) halo_bwd[i] = overlap[i];
    if (include_halo_fwd[i]) halo_fwd[i] = overlap[i];
  }
  setup_tensor_descriptor(desc, tensor, halo_fwd, halo_bwd);
}

template <typename Tensor>
inline void setup_tensor_descriptor(
    cudnnTensorDescriptor_t &desc,
    const Tensor &tensor,
    bool include_halo=true) {
  std::vector<bool> include_halo_array(tensor.get_num_dims(), include_halo);
  setup_tensor_descriptor(desc, tensor, include_halo_array,
                          include_halo_array);
}

inline int get_tensor_dimension(const cudnnTensorDescriptor_t &desc, int d) {
  cudnnDataType_t dt;
  int dims[nb_dims_requested];
  int strides[nb_dims_requested];
  int nbdims;
  DISTCONV_CHECK_CUDNN(
      cudnnGetTensorNdDescriptor(desc, nb_dims_requested, &dt, &nbdims, dims, strides));
  assert_always(d < nbdims);
  return dims[nbdims-d-1];
}

inline void set_tensor_dimension(cudnnTensorDescriptor_t &desc,
                                 int d, int n) {
  cudnnDataType_t dt;
  int dims[nb_dims_requested];
  int strides[nb_dims_requested];
  int nbdims;
  DISTCONV_CHECK_CUDNN(
      cudnnGetTensorNdDescriptor(desc, nb_dims_requested, &dt, &nbdims, dims, strides));
  assert_always(d < nbdims);
  dims[nbdims-d-1] = n;
  DISTCONV_CHECK_CUDNN(
      cudnnSetTensorNdDescriptor(desc, dt, nbdims, dims, strides));
}

inline int get_tensor_num_dimensions(const cudnnTensorDescriptor_t &desc) {
  cudnnDataType_t dt;
  int nbdims;
  DISTCONV_CHECK_CUDNN(
      cudnnGetTensorNdDescriptor(desc, 0, &dt, &nbdims, nullptr, nullptr));
  return nbdims;
}

inline void set_tensor_num_samples(cudnnTensorDescriptor_t &desc,
                                   int n) {
  int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
  set_tensor_dimension(desc, num_sample_dim, n);
}

inline int get_tensor_num_samples(const cudnnTensorDescriptor_t &desc) {
  int num_sample_dim = get_tensor_num_dimensions(desc) - 1;
  return get_tensor_dimension(desc, num_sample_dim);
}

inline void copy_tensor_descriptor(cudnnTensorDescriptor_t &dst,
                                   const cudnnTensorDescriptor_t &src) {
  cudnnDataType_t dt;
  int dims[nb_dims_requested];
  int strides[nb_dims_requested];
  int nbdims;
  DISTCONV_CHECK_CUDNN(
      cudnnGetTensorNdDescriptor(src, nb_dims_requested, &dt, &nbdims, dims, strides));
  DISTCONV_CHECK_CUDNN(
      cudnnSetTensorNdDescriptor(dst, dt, nbdims, dims, strides));
}

inline void copy_filter_descriptor(cudnnFilterDescriptor_t &dst,
                                   const cudnnFilterDescriptor_t &src) {
  cudnnDataType_t dt;
  int dims[nb_dims_requested];
  int nbdims;
  cudnnTensorFormat_t fmt;
  DISTCONV_CHECK_CUDNN(
      cudnnGetFilterNdDescriptor(src, nb_dims_requested, &dt, &fmt, &nbdims, dims));
  DISTCONV_CHECK_CUDNN(
      cudnnSetFilterNdDescriptor(dst, dt, fmt, nbdims, dims));
}

inline void copy_convolution_descriptor(
    cudnnConvolutionDescriptor_t &dst,
    const cudnnConvolutionDescriptor_t &src) {
  int array_length;
  const int arrayLengthRequested = 100;
  int pads[arrayLengthRequested];
  int strides[arrayLengthRequested];
  int dilations[arrayLengthRequested];
  cudnnConvolutionMode_t mode;
  cudnnDataType_t dt;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(
      src, arrayLengthRequested, &array_length, pads, strides, dilations, &mode, &dt));
  DISTCONV_CHECK_CUDNN(cudnnSetConvolutionNdDescriptor(
      dst, array_length,
      pads, strides,
      dilations, mode, dt));
}

inline void copy_pooling_descriptor(
    cudnnPoolingDescriptor_t &dst,
    const cudnnPoolingDescriptor_t &src) {
  cudnnPoolingMode_t mode;
  cudnnNanPropagation_t nan_prop;
  int ndims;
  int window_dims[nb_dims_requested];
  int padding[nb_dims_requested];
  int strides[nb_dims_requested];

  DISTCONV_CHECK_CUDNN(
      cudnnGetPoolingNdDescriptor(src, nb_dims_requested, &mode, &nan_prop,
                                  &ndims, window_dims, padding,
                                  strides));
  DISTCONV_CHECK_CUDNN(
      cudnnSetPoolingNdDescriptor(dst, mode, nan_prop, ndims,
                                  window_dims, padding, strides));
}

inline void copy_activation_descriptor(
    cudnnActivationDescriptor_t &dst,
    const cudnnActivationDescriptor_t &src) {
  cudnnActivationMode_t mode;
  cudnnNanPropagation_t nan_prop;
  double coef;
  DISTCONV_CHECK_CUDNN(
      cudnnGetActivationDescriptor(src, &mode, &nan_prop, &coef));
  DISTCONV_CHECK_CUDNN(
      cudnnSetActivationDescriptor(dst, mode, nan_prop, coef));
}

struct Options {
  bool m_overlap_halo_exchange = false;
  bool m_deterministic = false;
  bool m_enable_profiling = false;
  float m_ws_capacity_factor = 1.0;
  Options(bool overlap_halo_exchange=false,
          bool deterministic=false,
          bool enable_profiling=false,
          bool ws_capacity_factor=1.0):
      m_overlap_halo_exchange(overlap_halo_exchange),
      m_deterministic(deterministic),
      m_enable_profiling(enable_profiling),
      m_ws_capacity_factor(ws_capacity_factor) {
    set_by_environment_variables();
  }
  void set_by_environment_variables() {
    if (std::getenv("DISTCONV_OVERLAP_HALO_EXCHANGE")) {
      util::MPIRootPrintStreamDebug() << "Environment variable: "
                                      << "DISTCONV_OVERLAP_HALO_EXCHANGE"
                                      << " detected";
      m_overlap_halo_exchange = true;
    }
    if (std::getenv("DISTCONV_DETERMINISTIC")) {
      util::MPIRootPrintStreamDebug() << "Environment variable: "
                                      << "DISTCONV_DETERMINISTIC"
                                      << " detected";
      m_deterministic = true;
    }
    if (std::getenv("DISTCONV_ENABLE_PROFILING")) {
      util::MPIRootPrintStreamDebug() << "Environment variable: "
                                      << "DISTCONV_ENABLE_PROFILING"
                                      << " detected";
      m_enable_profiling = true;
    }
    if (std::getenv("DISTCONV_WS_CAPACITY_FACTOR")) {
      util::MPIRootPrintStreamDebug() << "Environment variable: "
                                      << "DISTCONV_WS_CAPACITY_FACTOR"
                                      << " detected";
      m_ws_capacity_factor = atof(std::getenv("DISTCONV_WS_CAPACITY_FACTOR"));
    }
  }
};

// Backend context
class BackendCUDNN {
 public:
  BackendCUDNN(MPI_Comm comm, cudnnHandle_t cudnn_h,
               const Options &opts=Options()):
      m_cudnn_h(cudnn_h), m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
      m_p2p(comm),
#endif // DISTCONV_HAS_P2P
      m_opts(opts) {
    DISTCONV_CHECK_CUDA(cudaStreamCreate(&m_stream));
    init(comm);
  }

  BackendCUDNN(MPI_Comm comm, cudnnHandle_t cudnn_h,
               cudaStream_t stream, const Options &opts=Options()):
      m_cudnn_h(cudnn_h), m_stream(stream), m_enable_nvtx(false),
#ifdef DISTCONV_HAS_P2P
      m_p2p(comm),
#endif // DISTCONV_HAS_P2P
      m_opts(opts) {
    init(comm);
  }

  ~BackendCUDNN() {
#ifdef DISTCONV_HAS_P2P
    m_p2p.disconnect_all();
#endif // DISTCONV_HAS_P2P
  }

  std::string get_name() const {
    return std::string("CUDNN");
  }

  const Options &get_options() {
    return m_opts;
  }

  void wait() {
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(m_stream));
  }

  MPI_Comm get_comm() {
    return m_comm;
  }

  std::shared_ptr<Al::MPICUDABackend::comm_type> get_al_mpi_cuda_comm() {
    return m_al_mpi_cuda_comm;
  }

  Al::NCCLBackend::comm_type &get_al_nccl_comm() {
    return *m_al_nccl_comm;
  }

  cudnnHandle_t get_handle() {
    return m_cudnn_h;
  }

  cudaStream_t get_stream() {
    return m_stream;
  }

  void ensure_workspace(size_t size) {
    //util::PrintStreamDebug() << "Requested Workspace: " << size << "\n";
    if (m_ws.get_size() < size) {
      m_ws.allocate(size);
    }
    //util::PrintStreamDebug() << "Workspace: " << size << "\n";
  }

  void *get_workspace(size_t size) {
    ensure_workspace(size);
    return m_ws.get();
  }

  void enable_nvtx_marking(bool b=true) {
    m_enable_nvtx = b;
  }

  void disable_nvtx_marking() {
    enable_nvtx_marking(false);
  }

  bool is_nvtx_enabled() const {
    return m_enable_nvtx;
  }

#ifdef DISTCONV_HAS_P2P
  p2p::P2P &get_p2p() {
    return m_p2p;
  }
#endif // DISTCONV_HAS_P2P

  cudaStream_t get_internal_stream(int idx) {
    assert_always(idx < (int)m_internal_streams.size());
    return m_internal_streams[idx];
  }

  cudaStream_t get_internal_stream_pr(int idx) {
    assert_always(idx < (int)m_internal_streams_pr.size());
    return m_internal_streams_pr[idx];
  }

  std::shared_ptr<Al::MPICUDABackend::comm_type> &get_internal_al_mpi_cuda_comm(int idx) {
    assert_always(idx < (int)m_internal_streams_pr.size());
    return m_internal_al_mpi_cuda_comms[idx];
  }

  void wait_main_stream(int idx) {
    util::wait_stream(m_stream, get_internal_stream(idx));
  }

  void wait_main_stream_pr(int idx) {
    util::wait_stream(m_stream, get_internal_stream_pr(idx));
  }

  void wait_internal_stream(int idx) {
    util::wait_stream(
        get_internal_stream(idx), m_stream);
  }

  void wait_internal_stream_pr(int idx) {
    util::wait_stream(
        get_internal_stream_pr(idx), m_stream);
  }

  void sync_internal_stream(int idx) {
    util::sync_stream(m_stream, get_internal_stream(idx));
  }

  void sync_internal_stream_pr(int idx) {
    util::sync_stream(m_stream, get_internal_stream_pr(idx));
  }

  cudnnConvolutionFwdAlgo_t get_fwd_algorithm(
      const std::string &name,
      const cudnnTensorDescriptor_t *input_desc,
      const void *input,
      const cudnnFilterDescriptor_t *filter_desc,
      const void *filter,
      const cudnnConvolutionDescriptor_t *conv_desc,
      const cudnnTensorDescriptor_t *output_desc,
      void *output,
      size_t ws_size);

  cudnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm(
      const std::string &name,
      const cudnnFilterDescriptor_t *filter_desc,
      const void *filter,
      const cudnnTensorDescriptor_t *d_output_desc,
      const void *d_output,
      const cudnnConvolutionDescriptor_t *conv_desc,
      const cudnnTensorDescriptor_t *d_input_desc,
      void *d_input,
      size_t ws_size);

  cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm(
      const std::string &name,
      const cudnnTensorDescriptor_t *input_desc,
      const void *input,
      const cudnnTensorDescriptor_t *d_output_desc,
      const void *d_output,
      const cudnnConvolutionDescriptor_t *conv_desc,
      const cudnnFilterDescriptor_t *d_filter_desc,
      void *d_filter,
      size_t ws_size);

  void init_chanfilt_comm(index_t seg, MPI_Comm comm) {
    assert0(m_chanfilt_comms.count(seg));
    util::MPIPrintStreamDebug()
      << "Setting up new chanfilt comm for segments=" << seg;
    m_chanfilt_comms[seg] = std::unique_ptr<Al::NCCLBackend::comm_type>(
      new Al::NCCLBackend::comm_type(comm, get_stream()));
  }

  void init_segmented_ar_comm(index_t seg, MPI_Comm comm) {
    assert0(m_segmented_ar_comms.count(seg));
    util::MPIPrintStreamDebug()
      << "Setting up new segmented AR comm for segments=" << seg;
    m_segmented_ar_comms[seg] = std::unique_ptr<Al::NCCLBackend::comm_type>(
      new Al::NCCLBackend::comm_type(comm, get_stream()));
  }

  Al::NCCLBackend::comm_type *get_chanfilt_comm(index_t seg) {
    if (m_chanfilt_comms.count(seg) > 0) {
      return m_chanfilt_comms[seg].get();
    }
    return nullptr;
  }

  Al::NCCLBackend::comm_type *get_segmented_ar_comm(index_t seg) {
    if (m_segmented_ar_comms.count(seg) > 0) {
      return m_segmented_ar_comms[seg].get();
    }
    return nullptr;
  }

 protected:
  MPI_Comm m_comm;
  std::shared_ptr<Al::MPICUDABackend::comm_type> m_al_mpi_cuda_comm;
  // Keeps a heap object as copying a NCCLCommunicator destroys
  // ncclComm_t
  std::unique_ptr<Al::NCCLBackend::comm_type> m_al_nccl_comm;
  cudnnHandle_t m_cudnn_h;
  cudaStream_t m_stream;
  tensor::Memory<tensor::CUDAAllocator> m_ws;
  bool m_enable_nvtx;
#ifdef DISTCONV_HAS_P2P
  p2p::P2P m_p2p;
#endif // DISTCONV_HAS_P2P
  // the number of internal streams; should be larger than the number
  // of bounary planes
  static constexpr int m_num_internal_streams = 8;
  std::vector<cudaStream_t> m_internal_streams;
  static constexpr int m_num_internal_streams_pr = 8;
  std::vector<cudaStream_t> m_internal_streams_pr;
  // The communicator of MPICUDABackend creates new MPI communicators
  // when constructed even without no argument. Having them as heap
  // objects prevent that.
  std::vector<std::shared_ptr<Al::MPICUDABackend::comm_type>> m_internal_al_mpi_cuda_comms;
  Options m_opts;

  // Segmented communicators for channel/filter communication.
  std::unordered_map<index_t, std::unique_ptr<Al::NCCLBackend::comm_type>> m_chanfilt_comms;
  std::unordered_map<index_t, std::unique_ptr<Al::NCCLBackend::comm_type>> m_segmented_ar_comms;

  void init(MPI_Comm comm) {
    DISTCONV_CHECK_MPI(MPI_Comm_dup(comm, &m_comm));
    m_al_mpi_cuda_comm = std::make_shared<Al::MPICUDABackend::comm_type>(
        m_comm, m_stream);
    m_al_nccl_comm.reset(new Al::NCCLBackend::comm_type(m_comm, m_stream));
    DISTCONV_CHECK_CUDNN(cudnnSetStream(m_cudnn_h, m_stream));
    setup_internal_streams();
    setup_al_comms();
  }

  void setup_internal_streams() {
    for (int i = 0; i < m_num_internal_streams; ++i) {
      cudaStream_t s;
      DISTCONV_CHECK_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
      m_internal_streams.push_back(s);
    }
    for (int i = 0; i < m_num_internal_streams_pr; ++i) {
      m_internal_streams_pr.push_back(util::create_priority_stream());
    }
  }

  void setup_al_comms() {
    for (int i = 0; i < m_num_internal_streams_pr; ++i) {
      m_internal_al_mpi_cuda_comms.push_back(
          std::make_shared<Al::MPICUDABackend::comm_type>(m_comm, m_internal_streams_pr[i]));
    }
  }

  cudnnConvolutionFwdAlgo_t get_fwd_algorithm_by_heuristics(
      const cudnnTensorDescriptor_t &input_desc,
      const cudnnFilterDescriptor_t &filter_desc,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnTensorDescriptor_t &output_desc,
      size_t ws_size);

  cudnnConvolutionFwdAlgo_t autotune_fwd_algorithm(
      const cudnnTensorDescriptor_t &input_desc,
      const void *input,
      const cudnnFilterDescriptor_t &filter_desc,
      const void *filter,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnTensorDescriptor_t &output_desc,
      void *output,
      size_t ws_size);

  cudnnConvolutionBwdDataAlgo_t get_bwd_data_algorithm_by_heuristics(
      const cudnnFilterDescriptor_t &filter_desc,
      const cudnnTensorDescriptor_t &d_output_desc,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnTensorDescriptor_t &d_input_desc,
      size_t ws_size);

  cudnnConvolutionBwdDataAlgo_t autotune_bwd_data_algorithm(
      const cudnnFilterDescriptor_t &filter_desc,
      const void *filter,
      const cudnnTensorDescriptor_t &d_output_desc,
      const void *d_output,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnTensorDescriptor_t &d_input_desc,
      void *d_input,
      size_t ws_size);

  cudnnConvolutionBwdFilterAlgo_t get_bwd_filter_algorithm_by_heuristics(
      const cudnnTensorDescriptor_t &input_desc,
      const cudnnTensorDescriptor_t &d_output_desc,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnFilterDescriptor_t &d_filter_desc,
      size_t ws_size);

  cudnnConvolutionBwdFilterAlgo_t autotune_bwd_filter_algorithm(
      const cudnnTensorDescriptor_t &input_desc,
      const void *input,
      const cudnnTensorDescriptor_t &d_output_desc,
      const void *d_output,
      const cudnnConvolutionDescriptor_t &conv_desc,
      const cudnnFilterDescriptor_t &d_filter_desc,
      void *d_filter,
      size_t ws_size);
};

} // namespace cudnn
} // namespace distconv

#include "distconv/cudnn/convolution.hpp"
#include "distconv/cudnn/pooling.hpp"
#include "distconv/cudnn/relu.hpp"
#include "distconv/cudnn/leaky_relu.hpp"
#include "distconv/cudnn/batchnorm.hpp"
