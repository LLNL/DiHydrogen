#include "distconv/dnn_backend/pack_unpack.hpp"

#include "./dnn_lib_utils.hpp"

#include "distconv/util/util.hpp"
#include "h2/gpu/logger.hpp"
#include "h2/gpu/memory_utils.hpp"

// NOTE: This file has explicit dependency on cuDNN/MIOpen.
#if H2_HAS_CUDA
#include "distconv/util/util_cudnn.hpp"

#include <cudnn.h>
using Stream_t = cudaStream_t;

#elif H2_HAS_ROCM
#include "distconv/util/util_miopen.hpp"

#include <miopen/miopen.h>
using Stream_t = hipStream_t;

#endif

#include <numeric>
#include <stdexcept>
#include <variant>
#include <vector>

namespace distconv
{
// Declaration only; defined in the .cu file.
void do_gpu_tensor_repack(float const& alpha,
                          float const& beta,
                          size_t const ndims,
                          int const* dims,
                          int const* src_strides,
                          int const* tgt_strides,
                          float const* src_data,
                          float* tgt_data,
                          Stream_t stream);

// Utilities
namespace
{

// The behavior we should have is to just be able to shove whatever
// (valid) tensor we want through these interfaces. HOWEVER, doing
// this on ROCm platfmorms means accepting incorrect results, and this
// is not acceptible. The default behavior, therefore, is to "opt-in"
// on CUDA platforms and "opt-out" on ROCm platforms. In the code
// below, explicitly setting the environment variable will use the
// truthiness of the variable's value to determine whether to
// pack/unpack or just pass tensors through. Leaving the variable
// unset will pass tensors through on non-ROCm platforms and will
// pack/unpack on ROCm platforms.
bool do_pack_unpack() noexcept
{
    static bool const val = []() {
#if H2_HAS_ROCM
        bool tf = true;
#else
        bool tf = false;
#endif
        char const* env = std::getenv("H2_DISTCONV_FORCE_PACKED");
        if (env)
            tf = (env && std::strlen(env) && env[0] != '0');
        // Any nonempty string matching "[^0].*" is truthy.
        H2_GPU_DEBUG("Doing pack/unpack: {}", tf);
        return tf;
    }();

    return val;
}

bool is_fully_packed(std::vector<int> const& dims,
                     std::vector<int> const& strides)
{
    // As far as I know, LBANN doesn't do any overlapping striding
    // (this is exceptionally poorly supported in the real world and
    // it has semantic issues). Thus, a tensor is fully packed if and
    // only if strides[0] == prod(dims[1:]).
    return strides.front()
           == std::accumulate(std::next(dims.cbegin()),
                              dims.cend(),
                              1,
                              std::multiplies<int>{});
}

// std::tuple<cudnnDataType_t, std::vector<int>, std::vector<int>>
// but with nice names.
struct MyTensorDesc
{
    GPUDNNBackend::DataType_t dt;
    std::vector<int> dims;
    std::vector<int> strides;
    void set_ndims(size_t ndims)
    {
        dims.resize(ndims);
        strides.resize(ndims);
    }
    size_t memory_size() const
    {
        assert_eq(dims.size(), strides.size());
        assert_always(dims.size() > 0);
        return dims[0] * strides[0] * datatype_size(dt);
    }
};

MyTensorDesc get_details(GPUDNNBackend::TensorDescriptor_t desc)
{
    GPUDNNBackend::DataType_t dt;
    std::vector<int> dims, strides;
    GPUDNNBackend::get_tensor_descriptor(desc, dt, dims, strides);
    return {dt, std::move(dims), std::move(strides)};
};

GPUDNNBackend::TensorDescriptor_t make_backend_desc(MyTensorDesc my_desc)
{
    auto desc = GPUDNNBackend::make_tensor_descriptor();
    auto& [dt, dims, strides] = my_desc;
    GPUDNNBackend::set_tensor_descriptor(desc, dt, dims, strides);
    return desc;
}

// If the input tensor descriptor is already packed, then return it
// directly. Otherwise, create a new handle and set it up with the
// same dimensions but fully packed strides.
GPUDNNBackend::TensorDescriptor_t
get_packed_desc(GPUDNNBackend::TensorDescriptor_t desc)
{
    auto const [dt, dims, strides] = get_details(desc);
    if (is_fully_packed(dims, strides))
        return desc;
    else
        return make_backend_desc({dt, dims, get_fully_packed_strides(dims)});
}

struct MyTypeErasedPtr
{
    void* data;
    GPUDNNBackend::DataType_t dt;
    template <typename T, typename U>
    operator std::tuple<T, U>()
    {
        return {data, dt};
    }
};

MyTypeErasedPtr allocate(GPUDNNBackend::Handle_t handle,
                         GPUDNNBackend::TensorDescriptor_t desc)
{
    auto const [dt, dims, strides] = get_details(desc);
    auto const mem_size = dims[0] * strides[0] * datatype_size(dt);

    // Stream-aware allocation
    void* data;
    static_cast<void>(h2::gpu::default_cub_allocator().DeviceAllocate(
        &data, mem_size, GPUDNNBackend::get_stream(handle)));
    return {data, dt};
}

void copy_tensor(GPUDNNBackend::Handle_t handle,
                 host_scalar const& alpha,
                 GPUDNNBackend::TensorDescriptor_t src_desc,
                 void const* src_data,
                 host_scalar const& beta,
                 GPUDNNBackend::TensorDescriptor_t tgt_desc,
                 void* tgt_data)
{
#if H2_HAS_CUDA
    DISTCONV_CHECK_CUDNN(cudnnTransformTensor(
        handle, alpha, src_desc, src_data, beta, tgt_desc, tgt_data));
#elif H2_HAS_ROCM
    auto const stream = GPUDNNBackend::get_stream(handle);
    auto const [src_dt, src_dims, src_strides] = get_details(src_desc);
    auto const [tgt_dt, tgt_dims, tgt_strides] = get_details(tgt_desc);
    assert_always(src_dt == tgt_dt);
    assert_always(src_dims == tgt_dims);
    switch (src_dt)
    {
    case miopenFloat:
        do_gpu_tensor_repack(*reinterpret_cast<float const*>(alpha.get()),
                             *reinterpret_cast<float const*>(beta.get()),
                             src_dims.size(),
                             src_dims.data(),
                             src_strides.data(),
                             tgt_strides.data(),
                             reinterpret_cast<float const*>(src_data),
                             reinterpret_cast<float*>(tgt_data),
                             stream);
        break;
    default: throw std::runtime_error("Only float.");
    }
#endif
}

} // namespace

// Read proxy impl

PackedTensorReadProxy::PackedTensorReadProxy(
    GPUDNNBackend::TensorDescriptor_t unpacked_desc, bool const force)
    : m_unpacked_desc{unpacked_desc},
      m_packed_desc{unpacked_desc},
      m_unpacked_data{nullptr},
      m_packed_data{nullptr}
{
    if (force || do_pack_unpack())
        m_packed_desc = get_packed_desc(m_unpacked_desc);
}

PackedTensorReadProxy::PackedTensorReadProxy(
    GPUDNNBackend::Handle_t handle,
    GPUDNNBackend::TensorDescriptor_t unpacked_desc,
    void const* unpacked_data,
    bool const force)
    : m_unpacked_desc{unpacked_desc},
      m_packed_desc{unpacked_desc},
      m_unpacked_data{unpacked_data},
      m_packed_data{nullptr}
{
    if (force || do_pack_unpack())
        m_packed_desc = get_packed_desc(m_unpacked_desc);

    if (m_unpacked_desc == m_packed_desc)
        m_packed_data = const_cast<void*>(m_unpacked_data);
    else
    {
        GPUDNNBackend::DataType_t dt;
        std::tie(m_packed_data, dt) = allocate(handle, m_packed_desc);
        copy_tensor(handle,
                    make_host_scalar(dt, 1.0),
                    m_unpacked_desc,
                    m_unpacked_data,
                    make_host_scalar(dt, 0.0),
                    m_packed_desc,
                    m_packed_data);
    }
}

PackedTensorReadProxy::~PackedTensorReadProxy()
{
    if ((m_packed_data != m_unpacked_data) && m_packed_data)
    {
        static_cast<void>(
            h2::gpu::default_cub_allocator().DeviceFree(m_packed_data));
        m_packed_data = nullptr;
        m_unpacked_data = nullptr;
    }
    if (m_unpacked_desc != m_packed_desc)
        GPUDNNBackend::destroy_tensor_descriptor(m_packed_desc);
    m_packed_desc = 0;
    m_unpacked_desc = 0;
}

// Write proxy -- possibly copy in/copy out

PackedTensorWriteProxy::PackedTensorWriteProxy(
    GPUDNNBackend::TensorDescriptor_t unpacked_desc, bool const force)
    : m_unpacked_desc{unpacked_desc},
      m_packed_desc{unpacked_desc},
      m_unpacked_data{nullptr},
      m_packed_data{nullptr}
{
    if (force || do_pack_unpack())
        m_packed_desc = get_packed_desc(unpacked_desc);
}

PackedTensorWriteProxy::PackedTensorWriteProxy(
    GPUDNNBackend::Handle_t handle,
    GPUDNNBackend::TensorDescriptor_t unpacked_desc,
    void* unpacked_data,
    double beta,
    bool const force)
    : m_unpacked_desc{unpacked_desc},
      m_packed_desc{unpacked_desc},
      m_unpacked_data{unpacked_data},
      m_packed_data{nullptr},
      m_handle{handle}
{
    if (force || do_pack_unpack())
        m_packed_desc = get_packed_desc(unpacked_desc);

    // When "unpacked" == "packed", we don't actually need dt, so we
    // leave it as the default.
    if (m_unpacked_desc == m_packed_desc)
        m_packed_data = m_unpacked_data;
    else
    {
        std::tie(m_packed_data, m_dt) = allocate(m_handle, m_packed_desc);

        if (beta != 0.)
        {
            copy_tensor(m_handle,
                        make_host_scalar(m_dt, 1.0),
                        m_unpacked_desc,
                        m_unpacked_data,
                        make_host_scalar(m_dt, 0.0),
                        m_packed_desc,
                        m_packed_data);
        }
    }
}

// This is a "special" dtor because it can throw (and more
// importantly, semantically, it should be able to throw!). If we
// "unrolled" the code, this class replaces a pattern something like:
//
// x = make_writeable_proxy(unpacked_tensor);
// do_write_stuff(x);
// copy(x, unpacked_tensor)
//
// and we wouldn't terminate just because "copy(x, unpacked_tensor)"
// threw... It'd just be a normal exception that someone else could
// catch.
//
// However, the C++ rule is that a dtor cannot throw an exception
// while unwinding the stack to handle another exception -- such
// behavior guarantees an std::terminate. We can check for this,
// though, with std::uncaught_exceptions().
PackedTensorWriteProxy::~PackedTensorWriteProxy()
{
    if ((m_unpacked_data != m_packed_data) && m_packed_data)
    {
        if (!std::uncaught_exceptions())
        {
            copy_tensor(m_handle,
                        make_host_scalar(m_dt, 1.0),
                        m_packed_desc,
                        m_packed_data,
                        make_host_scalar(m_dt, 0.0),
                        m_unpacked_desc,
                        m_unpacked_data);
        }
        static_cast<void>(
            h2::gpu::default_cub_allocator().DeviceFree(m_packed_data));
        m_packed_data = nullptr;
        m_unpacked_data = nullptr;
    }
    if (m_unpacked_desc != m_packed_desc)
        GPUDNNBackend::destroy_tensor_descriptor(m_packed_desc);
    m_packed_desc = 0;
    m_unpacked_desc = 0;
}
} // namespace distconv
