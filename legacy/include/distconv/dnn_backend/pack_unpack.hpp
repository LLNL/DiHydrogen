#pragma once
#include <distconv_config.hpp>

#ifndef H2_LEGACY_INCLUDE_DISTCONV_CUDNN_BACKEND_HPP_INCLUDED
#error "Do not #include this file; just #include backend.hpp"
#endif

// Just need the typedefs for this file
#if H2_HAS_CUDA
#include <cudnn.h>
namespace distconv
{
namespace cudnn
{
using TensorDescriptor_t = cudnnTensorDescriptor_t;
using Handle_t = cudnnHandle_t;
using DataType_t = cudnnDataType_t;
}
}
#define H2_DNN_BACKEND_NS cudnn
#elif H2_HAS_ROCM
#include <miopen/miopen.h>
namespace distconv
{
namespace miopen
{
using TensorDescriptor_t = miopenTensorDescriptor_t;
using Handle_t = miopenHandle_t;
using DataType_t = miopenDataType_t;
}
}
#define H2_DNN_BACKEND_NS miopen
#endif

namespace distconv
{
namespace H2_DNN_BACKEND_NS
{

// This models a strided INPUT to a cuDNN operation. This object is
// simple: on construction, we allocate a buffer and copy the strided
// tensor into it. At destruction, we simply free the buffer (stack
// unwinding is irrelevant).
class PackedTensorReadProxy
{
    TensorDescriptor_t m_unpacked_desc = 0;
    TensorDescriptor_t m_packed_desc = 0;
    void const* m_unpacked_data = nullptr;
    void* m_packed_data = nullptr;

public:
    PackedTensorReadProxy(TensorDescriptor_t unpacked_desc);
    PackedTensorReadProxy(Handle_t handle,
                          TensorDescriptor_t unpacked_desc,
                          void const* unpacked_data);
    // This dtor can throw -- if the proxy code were "unrolled", this
    // would correspond to freeing any allocated memory/descriptors,
    // which can detect async errors and should be allowed to
    // (non-fatally) throw (don't ask whether any of our downstreams
    // are actually prepared to recover from such an error).
    ~PackedTensorReadProxy();

    // Direct access for those who know what they want
    TensorDescriptor_t unpacked_desc() const noexcept { return m_unpacked_desc; }
    TensorDescriptor_t packed_desc() const noexcept { return m_packed_desc; }
    void const* unpacked_data() const noexcept { return m_unpacked_data; }
    void const* packed_data() const noexcept { return m_packed_data; }

    // The "right" thing
    TensorDescriptor_t desc() const noexcept { return packed_desc(); }
    void const* ptr() const noexcept { return packed_data(); }

};// class PackedTensorReadProxy

// This models a strided output tensor. We need to allocate a packable
// buffer up front, then copy the values on destruction (as long as
// the stack is not unnaturally unwinding), and finally free the
// buffer. There's a potential complication when dealing with outputs
// of "accumulating operations": output = a * op(inputs) + b *
// output. I'm like 98% sure we only ever use a=1, b=0. HOWEVER, if b
// != 0, then we need to do some extra crap. There are 2 obvious
// options:
//
// 1. Copy the unpacked values into the packed buffer on construction,
//    copy the packed values into the unpacked buffer on destruction,
//    pass a'=a, b'=b to the backend call.
//
// 2. Allocate the packed buffer but do not initialize. Pass a'=1,
//    b'=0 to the backend call. Sum into the unpacked buffer at
//    destruction with "a * packed + b * unpacked" (this is similar to
//    what hipDNN does when a!=1,b!=0 anyway).
//
// Option 1 is less intrusive (doesn't require modifying the calling
// code's alpha, beta values). It might be more work but we can avoid
// it when b=0. (Also, cuDNN strongly encourages users to use beta=0
// whenever possible.)
class PackedTensorWriteProxy
{
    TensorDescriptor_t m_unpacked_desc = 0;
    TensorDescriptor_t m_packed_desc = 0;
    void* m_unpacked_data = nullptr;
    void* m_packed_data = nullptr;

    Handle_t m_handle;
    DataType_t m_dt;

public:
    PackedTensorWriteProxy(TensorDescriptor_t unpacked_desc);
    // Per the discussion above, this class will "copy on
    // construction" if beta!=0. If we switch to option 2 in the
    // future, we would need to add an alpha argument as well.
    PackedTensorWriteProxy(Handle_t handle,
                           TensorDescriptor_t unpacked_desc,
                           void* unpacked_data,
                           double beta = 0.);
    // This dtor can throw. See explanation/details below.
    ~PackedTensorWriteProxy();

    // Direct access for those who know what they want
    TensorDescriptor_t unpacked_desc() const noexcept { return m_unpacked_desc; }
    TensorDescriptor_t packed_desc() const noexcept { return m_packed_desc; }
    void const* unpacked_data() const noexcept { return m_unpacked_data; }
    void* packed_data() const noexcept { return m_packed_data; }

    // The "right" thing
    TensorDescriptor_t desc() const noexcept { return packed_desc(); }
    void* ptr() const noexcept { return packed_data(); }

};// class PackedTensorWriteProxy

inline PackedTensorReadProxy read_proxy(TensorDescriptor_t desc)
{
    return PackedTensorReadProxy{desc};
}

inline PackedTensorReadProxy read_proxy(Handle_t handle,
                                 TensorDescriptor_t desc,
                                 void const* data)
{
    return PackedTensorReadProxy{handle, desc, data};
}

inline PackedTensorWriteProxy write_proxy(TensorDescriptor_t desc)
{
    return PackedTensorWriteProxy{desc};
}

inline PackedTensorWriteProxy write_proxy(Handle_t handle,
                                          TensorDescriptor_t desc,
                                          void* data,
                                          double beta = 0.)
{
    return PackedTensorWriteProxy{handle, desc, data, beta};
}

}// namespace H2_DNN_BACKEND_NS
}// namespace distconv

#undef H2_DNN_BACKEND_NS
