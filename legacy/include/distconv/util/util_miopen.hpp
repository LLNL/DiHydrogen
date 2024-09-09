////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <ostream>
#include <type_traits>
#include <vector>

#include <miopen/miopen.h>

namespace distconv
{
namespace util
{

std::string get_name(miopenConvFwdAlgorithm_t const& algo);
std::string get_name(miopenConvBwdDataAlgorithm_t const& algo);
std::string get_name(miopenConvBwdWeightsAlgorithm_t const& algo);
std::string tostring(miopenTensorDescriptor_t const& desc);
std::string tostring(miopenConvolutionDescriptor_t const& desc);

std::ostream& operator<<(std::ostream& os, miopenDataType_t& dt);
inline std::ostream& operator<<(std::ostream& os,
                                miopenConvFwdAlgorithm_t const& algo)
{
  return os << get_name(algo);
}
inline std::ostream& operator<<(std::ostream& os,
                                miopenConvBwdDataAlgorithm_t const& algo)
{
  return os << get_name(algo);
}
inline std::ostream& operator<<(std::ostream& os,
                                miopenConvBwdWeightsAlgorithm_t const& algo)
{
  return os << get_name(algo);
}
inline std::ostream& operator<<(std::ostream& os,
                                miopenTensorDescriptor_t const& d)
{
  return os << tostring(d);
}
inline std::ostream& operator<<(std::ostream& os,
                                miopenConvolutionDescriptor_t const& d)
{
  return os << tostring(d);
}

// NOTE (trb 07/25/2022): There's no such thing in MIOpen.
// inline std::ostream& operator<<(std::ostream& os, cudnnTensorFormat_t& fmt)
// {
//     std::string fmt_string;
//     switch (fmt)
//     {
//     case CUDNN_TENSOR_NCHW: fmt_string = "NCHW"; break;
//     case CUDNN_TENSOR_NHWC: fmt_string = "NHWC"; break;
//     case CUDNN_TENSOR_NCHW_VECT_C: fmt_string = "NCHW_VECT_C"; break;
//     default: fmt_string = "UNKNOWN"; break;
//     }
//     return os << fmt_string;
// }

template <typename T>
struct miopenTypeTraits;

template <>
struct miopenTypeTraits<int>
{
  static constexpr auto value = miopenInt32; // Yeah, I know. But also, reality.
};

template <>
struct miopenTypeTraits<float>
{
  static constexpr auto value = miopenFloat;
};

template <>
struct miopenTypeTraits<double>
{
  static constexpr auto value = miopenDouble;
};

// FIXME (trb 07/25/2022): FP16 support.
// template <>
// struct miopenTypeTraits<__half>
// {
//     static constexpr value = miopenHalf;
// };

template <typename T>
inline constexpr miopenDataType_t get_miopen_type()
{
  return miopenTypeTraits<std::decay_t<T>>::value;
}

template <typename T>
inline constexpr miopenDataType_t miopen_type = get_miopen_type<T>();

template <typename T>
inline constexpr miopenDataType_t get_dnnlib_type()
{
  return get_miopen_type<T>();
}

/** @brief Get the loaded MIOpen version.
 *  @details This is a dynamic check that queries the library at runtime.
 */
std::string get_miopen_version_number_string();

/** @brief Create a dims vector compatible with MIOpen
 *  @details Assumes "NCHW" format.
 */
std::vector<int> get_miopen_dims(int const num_samples,
                                 int const num_channels,
                                 std::vector<int> const& spatial_dims);

/** @brief Create strides vector compatible with MIOpen.
 *  @details Only supports fully-packed NCHW format.
 */
std::vector<int> get_miopen_strides(int const num_samples,
                                    int const num_channels,
                                    std::vector<int> const& spatial_dims,
                                    std::string const& fmt);

} // namespace util
} // namespace distconv
