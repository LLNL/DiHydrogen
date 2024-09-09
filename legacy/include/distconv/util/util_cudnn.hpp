#pragma once

#include <utility>
#include <iostream>
#include <sstream>
#include <cudnn.h>
#include <cuda_fp16.h>

#include "distconv/util/util.hpp"
#include "distconv/util/util_cuda.hpp"

#define DISTCONV_CHECK_CUDNN(cudnn_call)                                \
  do {                                                                  \
    const cudnnStatus_t cudnn_status = cudnn_call;                      \
    if (cudnn_status != CUDNN_STATUS_SUCCESS) {                         \
      std::stringstream ss;                                             \
      ss << "cuDNN error: " << cudnnGetErrorString(cudnn_status)        \
         << std::endl << "Error at " << __FILE__ << ":" << __LINE__     \
         << std::endl;                                                  \
      std::cerr << ss.str();                                            \
      cudaDeviceReset();                                                \
      abort();                                                          \
    }                                                                   \
  } while (0)

namespace distconv {
namespace util {

inline std::ostream &operator<<(std::ostream &os, cudnnDataType_t &dt) {
  std::string s;
  switch(dt) {
    case CUDNN_DATA_FLOAT: s = "float"; break;
    case CUDNN_DATA_DOUBLE: s = "double"; break;
    case CUDNN_DATA_HALF: s = "half"; break;
    default: s = "UNKNOWN"; break;
  }
  return os << s;
}

struct CUDNNConvolutionFwdAlgorithms {
  const static int num = 10;
  using algo_pair = std::pair<cudnnConvolutionFwdAlgo_t, std::string>;
  algo_pair algos[num] = {
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                   "IMPLICIT_GEMM"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
                   "IMPLICIT_PRECOMP_GEMM"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
                   "GEMM"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
                   "DIRECT"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_FFT,
                   "FFT"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_FFT_TILING,
                   "FFT_TILING"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
                   "WINOGRAD"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED,
                   "WINOGRAD_NONFUSED"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                   "DEFAULT"),
    std::make_pair(CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
                   "DETERMINISTIC")};

  static int get_index(cudnnConvolutionFwdAlgo_t algo) {
    CUDNNConvolutionFwdAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].first == algo) return i;
    }
    return -1;
  }
  static int get_index(const std::string &name) {
    CUDNNConvolutionFwdAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].second == name) return i;
    }
    return -1;
  }
  static std::string get_name(cudnnConvolutionFwdAlgo_t algo) {
    CUDNNConvolutionFwdAlgorithms x;
    return x.algos[get_index(algo)].second;
  }
  static cudnnConvolutionFwdAlgo_t get_algo(const std::string &name) {
    CUDNNConvolutionFwdAlgorithms x;
    return x.algos[get_index(name)].first;
  }
  static std::string get_real_name(const std::string &name) {
    return get_name(get_algo(name));
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnConvolutionFwdAlgo_t &algo) {
  return os << CUDNNConvolutionFwdAlgorithms::get_name(algo);
}

inline std::string get_name(const cudnnConvolutionFwdAlgo_t &algo) {
  return CUDNNConvolutionFwdAlgorithms::get_name(algo);
}

struct CUDNNConvolutionBwdDataAlgorithms {
  const static int num = 8;
  using algo_pair = std::pair<cudnnConvolutionBwdDataAlgo_t, std::string>;
  algo_pair algos[num] = {
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                   "ALGO_0"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                   "ALGO_1"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT,
                   "FFT"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_FFT_TILING,
                   "FFT_TILING"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD,
                   "WINOGRAD"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED,
                   "WINOGRAD_NONFUSED"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_0,
                   "DEFAULT"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_DATA_ALGO_1,
                   "DETERMINISTIC")};

  static int get_index(cudnnConvolutionBwdDataAlgo_t algo) {
    CUDNNConvolutionBwdDataAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].first == algo) return i;
    }
    return -1;
  }
  static int get_index(const std::string &name) {
    CUDNNConvolutionBwdDataAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].second == name) return i;
    }
    return -1;
  }
  static std::string get_name(cudnnConvolutionBwdDataAlgo_t algo) {
    CUDNNConvolutionBwdDataAlgorithms x;
    return x.algos[get_index(algo)].second;
  }
  static cudnnConvolutionBwdDataAlgo_t get_algo(const std::string &name) {
    CUDNNConvolutionBwdDataAlgorithms x;
    return x.algos[get_index(name)].first;
  }
  static std::string get_real_name(const std::string &name) {
    return get_name(get_algo(name));
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnConvolutionBwdDataAlgo_t &algo) {
  return os << CUDNNConvolutionBwdDataAlgorithms::get_name(algo);
}

inline std::string get_name(const cudnnConvolutionBwdDataAlgo_t &algo) {
  return CUDNNConvolutionBwdDataAlgorithms::get_name(algo);
}

struct CUDNNConvolutionBwdFilterAlgorithms {
  const static int num = 9;
  using algo_pair = std::pair<cudnnConvolutionBwdFilterAlgo_t, std::string>;
  algo_pair algos[num] = {
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                   "ALGO_0"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                   "ALGO_1"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_3,
                   "ALGO_3"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT,
                   "FFT"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_FFT_TILING,
                   "FFT_TILING"),
    // This is not listed in the API document, but available in the
    // header file, and indeed can be chosen by the search method
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD,
                   "WINOGRAD"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED,
                   "WINOGRAD_NONFUSED"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0,
                   "DEFAULT"),
    std::make_pair(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1,
                   "DETERMINISTIC")};

  static int get_index(cudnnConvolutionBwdFilterAlgo_t algo) {
    CUDNNConvolutionBwdFilterAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].first == algo) return i;
    }
    return -1;
  }
  static int get_index(const std::string &name) {
    CUDNNConvolutionBwdFilterAlgorithms x;
    for (unsigned i = 0; i < x.num; ++i) {
      if (x.algos[i].second == name) return i;
    }
    return -1;
  }
  static std::string get_name(cudnnConvolutionBwdFilterAlgo_t algo) {
    CUDNNConvolutionBwdFilterAlgorithms x;
    int idx = get_index(algo);
    assert_always(idx != -1);
    return x.algos[idx].second;
  }
  static cudnnConvolutionBwdFilterAlgo_t get_algo(const std::string &name) {
    CUDNNConvolutionBwdFilterAlgorithms x;
    int idx = get_index(name);
    assert_always(idx != -1);
    return x.algos[idx].first;
  }
  static std::string get_real_name(const std::string &name) {
    return get_name(get_algo(name));
  }
};

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnConvolutionBwdFilterAlgo_t &algo) {
  return os << CUDNNConvolutionBwdFilterAlgorithms::get_name(algo);
}

inline std::string get_name(const cudnnConvolutionBwdFilterAlgo_t &algo) {
  return CUDNNConvolutionBwdFilterAlgorithms::get_name(algo);
}

inline std::ostream &operator<<(std::ostream &os, cudnnTensorFormat_t &fmt) {
  std::string fmt_string;
  switch(fmt) {
    case CUDNN_TENSOR_NCHW: fmt_string = "NCHW"; break;
    case CUDNN_TENSOR_NHWC: fmt_string = "NHWC"; break;
    case CUDNN_TENSOR_NCHW_VECT_C: fmt_string = "NCHW_VECT_C"; break;
    default: fmt_string = "UNKNOWN"; break;
  }
  return os << fmt_string;
}

inline std::string tostring(const cudnnTensorDescriptor_t &desc) {
  int req_num_dim = 8;
  cudnnDataType_t dt;
  int num_dims = 0;
  int dims[req_num_dim];
  int strides[req_num_dim];
  DISTCONV_CHECK_CUDNN(cudnnGetTensorNdDescriptor(
      desc, req_num_dim, &dt, &num_dims, dims, strides));
  std::stringstream ss;
  ss << "Tensor descriptor: #dims=" << num_dims;
  ss << ", type=" << dt;
  ss << ", dims=";
  for (int i = 0; i < num_dims; ++i) {
    ss << dims[i];
    if (i < num_dims - 1) ss << "x";
  }
  ss << ", strides=";
  for (int i = 0; i < num_dims; ++i) {
    ss << strides[i];
    if (i < num_dims - 1) ss << "x";
  }
  return ss.str();
}

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnTensorDescriptor_t &d) {
  return os << tostring(d);
}

inline std::string tostring(const cudnnFilterDescriptor_t &desc) {
  int req_num_dim = 8;
  cudnnDataType_t dt;
  cudnnTensorFormat_t fmt;
  int num_dims = 0;
  int dims[req_num_dim];
  DISTCONV_CHECK_CUDNN(cudnnGetFilterNdDescriptor(desc, req_num_dim, &dt, &fmt, &num_dims, dims));
  std::stringstream ss;
  ss << "Filter descriptor: format=" << fmt << ", #dims=" << num_dims;
  ss << ", type=" << dt;
  ss << ", dims=";
  for (int i = 0; i < num_dims; ++i) {
    ss << dims[i];
    if (i < num_dims - 1) ss << "x";
  }
  return ss.str();
}

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnFilterDescriptor_t &d) {
  return os << tostring(d);
}

inline std::string tostring(const cudnnConvolutionDescriptor_t &desc) {
  int req_array_length = 4;
  int array_length = 0;
  int pads[req_array_length];
  int strides[req_array_length];
  int dilations[req_array_length];
  cudnnConvolutionMode_t mode;
  cudnnDataType_t dt;
  DISTCONV_CHECK_CUDNN(cudnnGetConvolutionNdDescriptor(desc, req_array_length, &array_length,
                                                       pads, strides, dilations, &mode, &dt));
  std::stringstream ss;
  ss << "Convolution descriptor: array_length=" << array_length;
  ss << ", padding=";
  for (int i = 0; i < array_length; ++i) {
    ss << pads[i];
    if (i < array_length - 1) ss << "x";
  }
  ss << ", strides=";
  for (int i = 0; i < array_length; ++i) {
    ss << strides[i];
    if (i < array_length - 1) ss << "x";
  }
  ss << ", dilations=";
  for (int i = 0; i < array_length; ++i) {
    ss << dilations[i];
    if (i < array_length - 1) ss << "x";
  }
  ss << ", type=" << dt;
  return ss.str();
}

inline std::ostream &operator<<(std::ostream &os,
                                const cudnnConvolutionDescriptor_t &d) {
  return os << tostring(d);
}

template <typename T>
inline cudnnDataType_t get_cudnn_type();


template <>
inline cudnnDataType_t get_cudnn_type<float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
inline cudnnDataType_t get_cudnn_type<const float>() {
  return CUDNN_DATA_FLOAT;
}

template <>
inline cudnnDataType_t get_cudnn_type<double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
inline cudnnDataType_t get_cudnn_type<const double>() {
  return CUDNN_DATA_DOUBLE;
}

template <>
inline cudnnDataType_t get_cudnn_type<half>() {
  return CUDNN_DATA_HALF;
}

template <>
inline cudnnDataType_t get_cudnn_type<const half>() {
  return CUDNN_DATA_HALF;
}

template <typename T>
inline cudnnDataType_t get_dnnlib_type()
{
    return get_cudnn_type<T>();
}

inline std::string get_cudnn_version_number_string() {
  int version[3];
  cudnnGetProperty(MAJOR_VERSION, &version[0]);
  cudnnGetProperty(MINOR_VERSION, &version[1]);
  cudnnGetProperty(PATCH_LEVEL, &version[2]);
  std::stringstream ss;
  ss << "cuDNN v" << version[0] << "." << version[1] << "." << version[2];
  return ss.str();
}

inline std::vector<int> get_cudnn_dims(int num_samples, int num_channels,
                                       const std::vector<int> &spatial_dims) {
  std::vector<int> dims;
  dims.push_back(num_samples);
  dims.push_back(num_channels);
  dims.insert(dims.end(), spatial_dims.begin(), spatial_dims.end());
  return dims;
}

inline std::vector<int> get_cudnn_strides(int num_samples, int num_channels,
                                          const std::vector<int> &spatial_dims,
                                          const std::string &fmt) {
  int num_spatial_dims = spatial_dims.size();
  std::vector<int> strides(2+num_spatial_dims);
  assert_always(num_spatial_dims == 2 || num_spatial_dims == 3);
  if (fmt == "NCHW") {
    strides.back() = 1;
    auto sit = strides.rbegin();
    for (int i = num_spatial_dims - 1; i >= 0; --i) {
      *(sit + 1) = (*sit) * spatial_dims[i];
      ++sit;
    }
    *(sit + 1) = (*sit) * num_channels;
  } else {
    PrintStreamError() << "Unknown tensor format: " << fmt;
    std::abort();
  }
  return strides;
}

} // namespace util
} // namespace distconv
