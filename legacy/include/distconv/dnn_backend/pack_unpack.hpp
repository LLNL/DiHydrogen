#pragma once
#include <distconv_config.hpp>

#include "distconv/dnn_backend/dnn_backend.hpp"

namespace distconv
{

// This models a strided INPUT to a cuDNN/MIOpen operation. This
// object is simple: on construction, we allocate a buffer and copy
// the strided tensor into it. At destruction, we simply free the
// buffer (stack unwinding is irrelevant).
class PackedTensorReadProxy
{
    GPUDNNBackend::TensorDescriptor_t m_unpacked_desc = 0;
    GPUDNNBackend::TensorDescriptor_t m_packed_desc = 0;
    void const* m_unpacked_data = nullptr;
    void* m_packed_data = nullptr;
    GPUDNNBackend::Handle_t m_handle = nullptr;

public:
    /** @brief Construct a "descriptor-only" read proxy.
     *
     *  @param[in] unpacked_desc The descriptor for a (potentially)
     *                           non-packed tensor. If this is
     *                           actually packed, we just alias it. If
     *                           it's not packed, we create a
     *                           descriptor for a fully-packed tensor
     *                           of the same shape.
     *  @param[in] force If true, always proxy regardless of
     *                   the env var controls.
     */
    PackedTensorReadProxy(GPUDNNBackend::TensorDescriptor_t unpacked_desc,
                          bool force = false);

    /** @brief Construct a full read proxy.
     *
     *  A read proxy will pack the input data on construction, making
     *  it suitable for input to APIs that require packed tensors. The
     *  unpacked descriptor and data pointer are maintained for
     *  reference, but, strictly speaking, they are not needed by the
     *  internal mechanics of this proxy after this constructor
     *  exits. However, if the input tensor is actually packed
     *  already, the "packed_" APIs return aliases to the input
     *  descriptor and data.
     *
     *  @param[in] force If true, always proxy regardless of
     *                   the env var controls.
     */
    PackedTensorReadProxy(GPUDNNBackend::Handle_t handle,
                          GPUDNNBackend::TensorDescriptor_t unpacked_desc,
                          void const* unpacked_data,
                          bool force = false);
    // This dtor can throw -- if the proxy code were "unrolled", this
    // would correspond to freeing any allocated memory/descriptors,
    // which can detect async errors and should be allowed to
    // (non-fatally) throw (don't ask whether any of our downstreams
    // are actually prepared to recover from such an error).
    ~PackedTensorReadProxy();

    // Direct access for those who know what they want
    GPUDNNBackend::TensorDescriptor_t unpacked_desc() const noexcept
    {
        return m_unpacked_desc;
    }
    GPUDNNBackend::TensorDescriptor_t packed_desc() const noexcept
    {
        return m_packed_desc;
    }
    void const* unpacked_data() const noexcept { return m_unpacked_data; }
    void const* packed_data() const noexcept { return m_packed_data; }

    // The "right" thing
    GPUDNNBackend::TensorDescriptor_t desc() const noexcept
    {
        return packed_desc();
    }
    void const* ptr() const noexcept { return packed_data(); }

}; // class PackedTensorReadProxy

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
    GPUDNNBackend::TensorDescriptor_t m_unpacked_desc = 0;
    GPUDNNBackend::TensorDescriptor_t m_packed_desc = 0;
    void* m_unpacked_data = nullptr;
    void* m_packed_data = nullptr;

    GPUDNNBackend::Handle_t m_handle;
    GPUDNNBackend::DataType_t m_dt;

public:
    /** @brief Construct a "descriptor-only" read proxy.
     *
     *  @param[in] unpacked_desc The descriptor for a (potentially)
     *                           non-packed tensor. If this is
     *                           actually packed, we just alias it. If
     *                           it's not packed, we create a
     *                           descriptor for a fully-packed tensor
     *                           of the same shape.
     *  @param[in] force If true, always proxy regardless of
     *                   the env var controls.
     */
    PackedTensorWriteProxy(GPUDNNBackend::TensorDescriptor_t unpacked_desc,
                           bool force = false);
    // Per the discussion above, this class will "copy on
    // construction" if beta!=0. If we switch to option 2 in the
    // future, we would need to add an alpha argument as well.
    PackedTensorWriteProxy(GPUDNNBackend::Handle_t handle,
                           GPUDNNBackend::TensorDescriptor_t unpacked_desc,
                           void* unpacked_data,
                           double beta = 0.,
                           bool force = false);
    // This dtor can throw (whomp whomp). See explanation/details
    // below.
    ~PackedTensorWriteProxy();

    // Direct access for those who know what they want
    GPUDNNBackend::TensorDescriptor_t unpacked_desc() const noexcept
    {
        return m_unpacked_desc;
    }
    GPUDNNBackend::TensorDescriptor_t packed_desc() const noexcept
    {
        return m_packed_desc;
    }
    void const* unpacked_data() const noexcept { return m_unpacked_data; }
    void* packed_data() const noexcept { return m_packed_data; }

    // The "right" thing
    GPUDNNBackend::TensorDescriptor_t desc() const noexcept
    {
        return packed_desc();
    }
    void* ptr() const noexcept { return packed_data(); }

}; // class PackedTensorWriteProxy

inline PackedTensorReadProxy read_proxy(GPUDNNBackend::TensorDescriptor_t desc)
{
    return PackedTensorReadProxy{desc};
}

inline PackedTensorReadProxy read_proxy(GPUDNNBackend::Handle_t handle,
                                        GPUDNNBackend::TensorDescriptor_t desc,
                                        void const* data)
{
    return PackedTensorReadProxy{handle, desc, data};
}

inline PackedTensorWriteProxy
write_proxy(GPUDNNBackend::TensorDescriptor_t desc)
{
    return PackedTensorWriteProxy{desc};
}

inline PackedTensorWriteProxy
write_proxy(GPUDNNBackend::Handle_t handle,
            GPUDNNBackend::TensorDescriptor_t desc,
            void* data,
            double beta = 0.)
{
    return PackedTensorWriteProxy{handle, desc, data, beta};
}

// Force the proxies.
inline PackedTensorReadProxy
force_read_proxy(GPUDNNBackend::TensorDescriptor_t desc)
{
    return PackedTensorReadProxy{desc, /*force=*/true};
}

inline PackedTensorReadProxy
force_read_proxy(GPUDNNBackend::Handle_t handle,
                 GPUDNNBackend::TensorDescriptor_t desc,
                 void const* data)
{
    return PackedTensorReadProxy{handle, desc, data, /*force=*/true};
}

inline PackedTensorWriteProxy
force_write_proxy(GPUDNNBackend::TensorDescriptor_t desc)
{
    return PackedTensorWriteProxy{desc, /*force=*/true};
}

inline PackedTensorWriteProxy
force_write_proxy(GPUDNNBackend::Handle_t handle,
                  GPUDNNBackend::TensorDescriptor_t desc,
                  void* data,
                  double beta = 0.)
{
    return PackedTensorWriteProxy{handle, desc, data, beta, /*force=*/true};
}

} // namespace distconv
