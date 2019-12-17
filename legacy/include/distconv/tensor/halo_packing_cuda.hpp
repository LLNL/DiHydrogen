#pragma once

#include "distconv/base.hpp"
#include "distconv/tensor/halo_exchange_cuda.hpp"
#include "distconv/tensor/halo_cuda.hpp"

#ifdef DISTCONV_HAS_NVSHMEM
#include "distconv/util/nvshmem.hpp"
#endif // DISTCONV_HAS_NVSHMEM

#define HALO_EXCHANGE_ACCUME_OP_SWITCH(x)                       \
  switch(x) {                                                   \
    CASE_BLOCK(HaloExchangeAccumOp::ID);                        \
    CASE_BLOCK(HaloExchangeAccumOp::SUM);                       \
    CASE_BLOCK(HaloExchangeAccumOp::MAX);                       \
    CASE_BLOCK(HaloExchangeAccumOp::MIN);                       \
    default:                                                    \
      assert_always(0 && "Unknown accumulation op type");       \
  }

namespace distconv {
namespace tensor {
namespace halo_exchange_cuda {

template <typename DataType, HaloExchangeAccumOp op>
struct HaloExchangeAccumCUDAFunctor;

template <typename DataType>
struct HaloExchangeAccumCUDAFunctor<DataType,
                                    HaloExchangeAccumOp::ID> {
  __device__ void operator()(DataType &x, const DataType y) {
    x = y;
  }
};

template <typename DataType>
struct HaloExchangeAccumCUDAFunctor<DataType,
                                    HaloExchangeAccumOp::SUM> {
  __device__ void operator()(DataType &x, const DataType y) {
    x += y;
  }
};

template <typename DataType>
struct HaloExchangeAccumCUDAFunctor<DataType,
                                    HaloExchangeAccumOp::MAX> {
  __device__ void operator()(DataType &x, const DataType y) {
    x = util::max(x, y);
  }
};

template <typename DataType>
struct HaloExchangeAccumCUDAFunctor<DataType,
                                    HaloExchangeAccumOp::MIN> {
  __device__ void operator()(DataType &x, const DataType y) {
    x = util::min(x, y);
  }
};

template <typename DataType, bool pack, HaloExchangeAccumOp op>
struct PackFunctor {
  using Vec2 = typename util::GetVectorType<DataType, 2>::type;
  using Vec4 = typename util::GetVectorType<DataType, 4>::type;
  static constexpr HaloTraversalOpGroup group = HaloTraversalOpGroup::THREAD;
  static constexpr bool has_pre_grid = false;
  static constexpr bool has_post_grid = false;
  static constexpr bool modifies_tensor = true;
  DataType *m_buf;
  PackFunctor(DataType *buf): m_buf(buf) {}
  template <typename T> __device__
  typename std::enable_if<std::is_same<T, DataType>::value ||
                          std::is_same<T, Vec2>::value ||
                          std::is_same<T, Vec4>::value, void>::type
  operator()(T &x, size_t offset) {
    if (pack) {
      ((T*)m_buf)[offset] = x;
    } else {
      HaloExchangeAccumCUDAFunctor<T, op>()(
          x, ((T*)m_buf)[offset]);
    }
  }
};

template <typename DataType, bool is_pack, typename PackFunctor>
void pack_or_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse) {
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  TraverseHalo<TensorType, PackFunctor>(
      tensor, dim, side, width,
      (is_pack && !is_reverse) || (!is_pack && is_reverse),
      PackFunctor(static_cast<DataType*>(buf)), stream);
}

template <typename DataType, bool is_pack,
          template<typename, bool, HaloExchangeAccumOp> typename PackFunctor>
void pack_or_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse,
    HaloExchangeAccumOp op) {
#define CASE_BLOCK(OP)                                                  \
  case OP:                                                              \
    pack_or_unpack<DataType, is_pack,                                   \
                   PackFunctor<DataType, is_pack, OP>>(                 \
                       tensor, dim, side, width, stream, buf, is_reverse); \
    break;

  HALO_EXCHANGE_ACCUME_OP_SWITCH(op);
#undef CASE_BLOCK
}

template <typename DataType, bool is_pack>
void pack_or_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse,
    HaloExchangeAccumOp op) {
  pack_or_unpack<DataType, is_pack, PackFunctor>(
      tensor, dim, side, width, stream, buf, is_reverse, op);
}

template <typename DataType>
void pack_or_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_pack, bool is_reverse,
    HaloExchangeAccumOp op) {
  if (width == 0) return;
  if (is_pack) {
    pack_or_unpack<DataType, true>(
        tensor, dim, side, width, stream,
        buf, is_reverse, op);
  } else {
    pack_or_unpack<DataType, false>(
        tensor, dim, side, width, stream,
        buf, is_reverse, op);
  }
  return;
}

#ifdef DISTCONV_HAS_NVSHMEM

template <typename DataType>
struct PackAndPutBlockFunctor {
  using Vec2 = typename util::GetVectorType<DataType, 2>::type;
  using Vec4 = typename util::GetVectorType<DataType, 4>::type;
  static constexpr HaloTraversalOpGroup group = HaloTraversalOpGroup::BLOCK;
  static constexpr bool has_pre_grid = false;
  static constexpr bool has_post_grid = false;
  static constexpr bool modifies_tensor = false;
  DataType *m_buf;
  DataType *m_dst;
  int m_peer;
  PackAndPutBlockFunctor(DataType *buf, DataType *dst, int peer):
      m_buf(buf), m_dst(dst), m_peer(peer) {}

  template <typename T> __device__
  typename std::enable_if<std::is_same<T, DataType>::value ||
                          std::is_same<T, Vec2>::value ||
                          std::is_same<T, Vec4>::value, void>::type
  operator()(const T &x, size_t offset_base, int thread_offset, int num_elms) {
    if (thread_offset < num_elms) {
      ((T*)(m_buf))[offset_base + thread_offset] = x;
    }
    util::nvshmem::put_nbi_block(((T*)m_dst) + offset_base,
                                 ((T*)m_buf) + offset_base,
                                 num_elms, m_peer);
  }
};

template <typename DataType>
void pack_and_put_block(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse, void *dst, int peer) {
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  TraverseHalo<TensorType, PackAndPutBlockFunctor<DataType>>(
      tensor, dim, side, width, !is_reverse,
      PackAndPutBlockFunctor<DataType>(static_cast<DataType*>(buf),
                                       static_cast<DataType*>(dst), peer), stream);
}

template <typename DataType>
struct PackPutNotifyBlockFunctor {
  using Vec2 = typename util::GetVectorType<DataType, 2>::type;
  using Vec4 = typename util::GetVectorType<DataType, 4>::type;
  static constexpr HaloTraversalOpGroup group = HaloTraversalOpGroup::BLOCK;
  static constexpr bool has_pre_grid = false;
  static constexpr bool has_post_grid = true;
  static constexpr bool modifies_tensor = false;
  DataType *m_buf;
  DataType *m_dst;
  int m_peer;
  util::nvshmem::PairwiseSyncDevice m_sync;
  PackPutNotifyBlockFunctor(DataType *buf, DataType *dst, int peer,
                            util::nvshmem::PairwiseSync &sync):
      m_buf(buf), m_dst(dst), m_peer(peer), m_sync(sync.get_for_device()) {}

  template <typename T> __device__
  typename std::enable_if<std::is_same<T, DataType>::value ||
                          std::is_same<T, Vec2>::value ||
                          std::is_same<T, Vec4>::value, void>::type
  operator()(const T &x, size_t offset_base, int thread_offset, int num_elms) {
    if (thread_offset < num_elms) {
      ((T*)(m_buf))[offset_base + thread_offset] = x;
    }
    util::nvshmem::put_nbi_block(((T*)m_dst) + offset_base,
                                 ((T*)m_buf) + offset_base,
                                 num_elms, m_peer);
  }

  __device__ void post() {
    m_sync.inc_counter();
    m_sync.notify(m_peer, util::nvshmem::SyncType::FENCE);
  }
};

template <typename DataType>
void pack_put_notify_block(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse, void *dst, int peer,
    util::nvshmem::PairwiseSync &sync) {
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  TraverseHalo<TensorType, PackPutNotifyBlockFunctor<DataType>>(
      tensor, dim, side, width, !is_reverse,
      PackPutNotifyBlockFunctor<DataType>(static_cast<DataType*>(buf),
                                          static_cast<DataType*>(dst),
                                          peer, sync),
      stream);
}

template <typename DataType, HaloExchangeAccumOp op>
struct WaitAndUnpackFunctor {
  using Vec2 = typename util::GetVectorType<DataType, 2>::type;
  using Vec4 = typename util::GetVectorType<DataType, 4>::type;
  static constexpr HaloTraversalOpGroup group = HaloTraversalOpGroup::THREAD;
  static constexpr bool has_pre_grid = true;
  static constexpr bool has_post_grid = false;
  static constexpr bool modifies_tensor = true;
  DataType *m_buf;
  util::nvshmem::PairwiseSyncDevice m_sync;
  WaitAndUnpackFunctor(DataType *buf, util::nvshmem::PairwiseSync &sync):
      m_buf(buf), m_sync(sync.get_for_device()) {}

  template <typename T> __device__
  typename std::enable_if<std::is_same<T, DataType>::value ||
                          std::is_same<T, Vec2>::value ||
                          std::is_same<T, Vec4>::value, void>::type
  operator()(T &x, size_t offset) {
    HaloExchangeAccumCUDAFunctor<T, op>()(
        x, ((T*)(m_buf))[offset]);
  }

  __device__ void pre() {
    m_sync.wait();
  }
};

template <typename DataType, HaloExchangeAccumOp Op>
void wait_and_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse, util::nvshmem::PairwiseSync &sync) {
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
  TraverseHalo<TensorType, WaitAndUnpackFunctor<DataType, Op>>(
      tensor, dim, side, width, is_reverse,
      WaitAndUnpackFunctor<DataType, Op>(static_cast<DataType*>(buf), sync),
      stream);
}

template <typename DataType>
void wait_and_unpack(
    Tensor<DataType, LocaleMPI, CUDAAllocator> &tensor,
    int dim, Side side, int width, cudaStream_t stream,
    void *buf, bool is_reverse, HaloExchangeAccumOp op,
    util::nvshmem::PairwiseSync &sync) {
#define CASE_BLOCK(OP)                                                  \
  case OP:                                                              \
    wait_and_unpack<DataType, OP>(                                      \
        tensor, dim, side, width, stream, buf, is_reverse, sync);       \
    break;

  HALO_EXCHANGE_ACCUME_OP_SWITCH(op);
#undef CASE_BLOCK
}

#undef HALO_EXCHANGE_ACCUME_OP_SWITCH

#endif // DISTCONV_HAS_NVSHMEM

} // namespace halo_exchange_cuda
} // namespace tensor
} // namespace distconv
