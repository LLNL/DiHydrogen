#pragma once

#include "distconv/util/util.hpp"
#include "distconv/util/util_rocm.hpp"

#include <iostream>
#include <map>
#include <sstream>
#include <utility>

#include <miopen/miopen.h>

#define DISTCONV_CHECK_MIOPEN(miopen_call)                                   \
    do                                                                       \
    {                                                                        \
        miopenStatus_t const status_distconv_check_miopen = (miopen_call);   \
        if (status_distconv_check_miopen != miopenStatusSuccess)             \
        {                                                                    \
            ::distconv::util::PrintStreamError()                             \
                << "MIOpen error at " << __FILE__ << ":" << __LINE__ << ": " \
                << miopenGetErrorString(status_distconv_check_miopen)        \
                << std::endl;                                                \
            static_cast<void>(hipDeviceReset());                             \
            abort();                                                         \
        }                                                                    \
    } while (0)

namespace distconv
{
namespace util
{

inline std::ostream& operator<<(std::ostream& os, miopenDataType_t& dt)
{
    switch (dt)
    {
    case miopenHalf: return os << "half";         // Fully supported
    case miopenFloat: return os << "float";       // Fully supported
    case miopenInt32: return os << "int32";       // Partially supported
    case miopenInt8: return os << "int8";         // Partially supported
    case miopenInt8x4: return os << "int8x4";     // Partially supported
    case miopenBFloat16: return os << "bfloat16"; // Partially supported
    case miopenDouble: return os << "double";     // Partially supported
    default: return os << "UNKNOWN";
    }
}

// FIXME (trb 07/25/2022): Need to setup "DEFAULT" and "DETERMINISTIC"
// for each of these.
struct MIOpenConvolutionFwdAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvFwdAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvFwdAlgorithm_t const algo) noexcept
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvFwdAlgorithm_t get_algo(std::string const& name) noexcept
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name) noexcept
    {
        return get_name(get_algo(name));
    }
};
inline MIOpenConvolutionFwdAlgorithms::map_type const
    MIOpenConvolutionFwdAlgorithms::algo_map = {
        {miopenConvolutionFwdAlgoGEMM, "GEMM"},
        {miopenConvolutionFwdAlgoDirect, "DIRECT"},
        {miopenConvolutionFwdAlgoFFT, "FFT"},
        {miopenConvolutionFwdAlgoWinograd, "WINOGRAD"},
        {miopenConvolutionFwdAlgoImplicitGEMM, "IMPLICIT_GEMM"},
};

inline std::ostream&
operator<<(std::ostream& os, miopenConvFwdAlgorithm_t const& algo)
{
    return os << MIOpenConvolutionFwdAlgorithms::get_name(algo);
}

inline std::string get_name(miopenConvFwdAlgorithm_t const& algo)
{
    return MIOpenConvolutionFwdAlgorithms::get_name(algo);
}

struct MIOpenConvolutionBwdDataAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvBwdDataAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvBwdDataAlgorithm_t algo)
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvBwdDataAlgorithm_t get_algo(std::string const& name)
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name)
    {
        return get_name(get_algo(name));
    }
};
inline MIOpenConvolutionBwdDataAlgorithms::map_type const
    MIOpenConvolutionBwdDataAlgorithms::algo_map = {
        {miopenConvolutionBwdDataAlgoGEMM, "GEMM"},
        {miopenConvolutionBwdDataAlgoDirect, "DIRECT"},
        {miopenConvolutionBwdDataAlgoFFT, "FFT"},
        {miopenConvolutionBwdDataAlgoWinograd, "WINOGRAD"},
        {miopenTransposeBwdDataAlgoGEMM, "TRANSPOSE GEMM - DEPRECATED"},
        {miopenConvolutionBwdDataAlgoImplicitGEMM, "IMPLICIT_GEMM"},
};

inline std::ostream&
operator<<(std::ostream& os, miopenConvBwdDataAlgorithm_t const& algo)
{
    return os << MIOpenConvolutionBwdDataAlgorithms::get_name(algo);
}

inline std::string get_name(miopenConvBwdDataAlgorithm_t const& algo)
{
    return MIOpenConvolutionBwdDataAlgorithms::get_name(algo);
}

struct MIOpenConvolutionBwdFilterAlgorithms
{
    using map_type =
        std::vector<std::pair<miopenConvBwdWeightsAlgorithm_t, std::string>>;
    static map_type const algo_map;
    static std::string get_name(miopenConvBwdWeightsAlgorithm_t algo)
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&algo](auto const& v) { return v.first == algo; });
        assert_always(iter != cend(algo_map));
        return iter->second;
    }
    static miopenConvBwdWeightsAlgorithm_t get_algo(std::string const& name)
    {
        auto const iter = std::find_if(
            cbegin(algo_map), cend(algo_map),
            [&name](auto const& v) { return v.second == name; });
        assert_always(iter != cend(algo_map));
        return iter->first;
    }
    static std::string get_real_name(std::string const& name)
    {
        return get_name(get_algo(name));
    }
};
inline MIOpenConvolutionBwdFilterAlgorithms::map_type const
    MIOpenConvolutionBwdFilterAlgorithms::algo_map = {
        {miopenConvolutionBwdWeightsAlgoGEMM, "GEMM"},
        {miopenConvolutionBwdWeightsAlgoDirect, "DIRECT"},
        {miopenConvolutionBwdWeightsAlgoDirect, "WINOGRAD"},
        {miopenConvolutionBwdWeightsAlgoDirect, "IMPLICIT_GEMM"},
};

inline std::ostream&
operator<<(std::ostream& os, miopenConvBwdWeightsAlgorithm_t const& algo)
{
    return os << MIOpenConvolutionBwdFilterAlgorithms::get_name(algo);
}

inline std::string get_name(miopenConvBwdWeightsAlgorithm_t const& algo)
{
    return MIOpenConvolutionBwdFilterAlgorithms::get_name(algo);
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

inline std::string tostring(miopenTensorDescriptor_t const& desc)
{
    int num_dims = -1;
    // This API is TERRIBLY named. This actually gets the number of
    // dimensions in the tensor.
    DISTCONV_CHECK_MIOPEN(miopenGetTensorDescriptorSize(desc, &num_dims));

    miopenDataType_t dt;
    std::vector<int> dims, strides;
    dims.reserve(num_dims);
    strides.reserve(num_dims);

    DISTCONV_CHECK_MIOPEN(
        miopenGetTensorDescriptor(desc, &dt, dims.data(), strides.data()));
    std::ostringstream oss;
    oss << "Tensor descriptor: #dims=" << num_dims;
    oss << ", type=" << dt;
    oss << ", dims=";
    for (int i = 0; i < num_dims; ++i)
        oss << dims[i] << (i < num_dims - 1 ? "x" : "");
    oss << ", strides=";
    for (int i = 0; i < num_dims; ++i)
        oss << strides[i] << (i < num_dims - 1 ? "x" : "");
    return oss.str();
}

inline std::ostream&
operator<<(std::ostream& os, miopenTensorDescriptor_t const& d)
{
    return os << tostring(d);
}

inline std::string tostring(miopenConvolutionDescriptor_t const& desc)
{
    int const max_spatial_dims = 5; // MIOpen supports up to 5-D tensors.
    std::vector<int> padding, strides, dilations;
    miopenConvolutionMode_t mode;
    padding.reserve(max_spatial_dims);
    strides.reserve(max_spatial_dims);
    dilations.reserve(max_spatial_dims);

    int spatial_dims = -1;
    // This gets the correct value for spatial_dims.
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(
        desc, 0, &spatial_dims, padding.data(), strides.data(),
        dilations.data(), &mode));
    assert_always(spatial_dims > 0 && spatial_dims <= 5);

    // This fills in the rest of the values.
    DISTCONV_CHECK_MIOPEN(miopenGetConvolutionNdDescriptor(
        desc, spatial_dims, &spatial_dims, padding.data(), strides.data(),
        dilations.data(), &mode));

    std::ostringstream oss;
    oss << "Convolution descriptor: spatial_dims=" << spatial_dims;
    oss << ", padding=";
    for (int i = 0; i < spatial_dims; ++i)
        oss << padding[i] << (i < spatial_dims - 1 ? "x" : "");
    oss << ", strides=";
    for (int i = 0; i < spatial_dims; ++i)
        oss << strides[i] << (i < spatial_dims - 1 ? "x" : "");
    oss << ", dilations=";
    for (int i = 0; i < spatial_dims; ++i)
        oss << dilations[i] << (i < spatial_dims - 1 ? "x" : "");
    return oss.str();
}

inline std::ostream&
operator<<(std::ostream& os, miopenConvolutionDescriptor_t const& d)
{
    return os << tostring(d);
}

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

inline std::string get_miopen_version_number_string()
{
    size_t version[3];
    DISTCONV_CHECK_MIOPEN(miopenGetVersion(&version[0], &version[1], &version[2]));
    std::ostringstream oss;
    oss << "MIOpen v" << version[0] << "." << version[1] << "." << version[2];
    return oss.str();
}

// "NCHW" format.
inline std::vector<int> get_miopen_dims(
    int const num_samples,
    int const num_channels,
    std::vector<int> const& spatial_dims)
{
    std::vector<int> dims;
    dims.reserve(spatial_dims.size() + 2);
    dims.push_back(num_samples);
    dims.push_back(num_channels);
    dims.insert(dims.end(), spatial_dims.begin(), spatial_dims.end());
    return dims;
}

// Only supports NCHW format; assumes fully-packed tensor.
inline std::vector<int> get_miopen_strides(
    int const num_samples,
    int const num_channels,
    std::vector<int> const& spatial_dims,
    std::string const& fmt)
{
    size_t const num_spatial_dims = spatial_dims.size();
    assert_always(num_spatial_dims == 2 || num_spatial_dims == 3);

    std::vector<int> strides(2 + num_spatial_dims, 1);
    if (fmt == "NCHW")
    {
        strides.back() = 1;
        auto sit = strides.rbegin();
        for (size_t i = num_spatial_dims - 1; i >= 0; --i)
        {
            *(sit + 1) = (*sit) * spatial_dims[i];
            ++sit;
        }
        *(sit + 1) = (*sit) * num_channels;
    }
    else
    {
        PrintStreamError() << "Unknown tensor format: " << fmt;
        std::abort();
    }
    return strides;
}

} // namespace util
} // namespace distconv
