#pragma once

#include "distconv/tensor/runtime_cuda.hpp"
#include "distconv/runtime_cuda.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/tensor/memory_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#include "distconv/util/util_cuda.hpp"

namespace distconv {
namespace tensor {

template <typename DataType>
class TensorMPICUDAShuffler {
 protected:
  using TensorType = Tensor<DataType, LocaleMPI, CUDAAllocator>;
 public:

  TensorMPICUDAShuffler(const TensorType &src_tensor,
                        const TensorType &dst_tensor,
                        DataType *src_buf=nullptr,
                        DataType *dst_buf=nullptr):
      m_src_local_shape(src_tensor.get_local_shape()),
      m_dst_local_shape(dst_tensor.get_local_shape()),
      m_src_strides(src_tensor.get_strides()),
      m_dst_strides(dst_tensor.get_strides()),
      m_src_locale_shape(src_tensor.get_locale_shape()),
      m_dst_locale_shape(dst_tensor.get_locale_shape()),
      m_src_overlap(src_tensor.get_overlap()),
      m_dst_overlap(dst_tensor.get_overlap()),
      m_loc(src_tensor.get_locale()),
      m_src_split_root(src_tensor.is_split_root()),
      m_dst_split_root(dst_tensor.is_split_root()),
      m_rank_limits_fwd(nullptr),
      m_rank_limits_bwd(nullptr),
      m_send_counts(nullptr), m_recv_counts(nullptr),
      m_send_displs_h(nullptr), m_recv_displs_h(nullptr),
      m_send_displs_d(nullptr), m_recv_displs_d(nullptr),
      m_src_buf(src_buf), m_dst_buf(dst_buf),
      m_src_buf_passed(src_buf != nullptr), m_dst_buf_passed(dst_buf != nullptr) {
    setup_rank_limits(src_tensor, dst_tensor, m_rank_limits_fwd);
    setup_rank_limits(dst_tensor, src_tensor, m_rank_limits_bwd);
    setup_displs(src_tensor, dst_tensor);

    int num_ranks = m_loc.get_size();
    for (int pid = 0; pid < num_ranks; ++pid) {
      if (m_send_counts[pid] != 0 || m_recv_counts[pid] != 0) {
        util::MPIPrintStreamDebug()
            << "Send/recv counts for "
            << pid << ": " << m_send_counts[pid] << ", "
            << m_recv_counts[pid];
        m_peers.push_back(pid);
      }
    }
  }

  virtual ~TensorMPICUDAShuffler() {
    if (m_rank_limits_fwd)
      DISTCONV_CHECK_CUDA(cudaFree(m_rank_limits_fwd));
    if (m_rank_limits_bwd)
      DISTCONV_CHECK_CUDA(cudaFree(m_rank_limits_bwd));
    if (m_send_counts)
      delete[] m_send_counts;
    if (m_recv_counts)
      delete[] m_recv_counts;
    if (m_send_displs_h)
      delete[] m_send_displs_h;
    if (m_recv_displs_h)
      delete[] m_recv_displs_h;
    if (m_send_displs_d)
      DISTCONV_CHECK_CUDA(cudaFree(m_send_displs_d));
    if (m_recv_displs_d)
      DISTCONV_CHECK_CUDA(cudaFree(m_recv_displs_d));
  }

  void shuffle_forward(const DataType *src, DataType *dst,
                       cudaStream_t stream=0);
  void shuffle_backward(const DataType *src, DataType *dst,
                        cudaStream_t stream=0);

  static size_t get_buf_size(const TensorType &tensor) {
    return get_buf_size(tensor.get_local_shape());
  }

  static size_t get_buf_size(const Shape &tensor_local_shape) {
    return tensor_local_shape.get_size() * sizeof(DataType);
  }

 protected:
  const Shape m_src_local_shape;
  const Shape m_dst_local_shape;
  const IndexVector m_src_strides;
  const IndexVector m_dst_strides;
  const Shape m_src_locale_shape;
  const Shape m_dst_locale_shape;
  const IntVector m_src_overlap;
  const IntVector m_dst_overlap;
  const LocaleMPI &m_loc;
  const bool m_src_split_root;
  const bool m_dst_split_root;

  // Offsets in src tensor for each dst locale. Used in
  // packing. Linearized to a 1D array.
  int *m_rank_limits_fwd;
  // Offsets in dst tensor for each src locale. Used in
  // packing. Linearized to a 1D array.
  int *m_rank_limits_bwd;
  int *m_send_counts;
  int *m_recv_counts;
  int *m_send_displs_h;
  int *m_recv_displs_h;
  int *m_send_displs_d;
  int *m_recv_displs_d;

  DataType *m_src_buf;
  DataType *m_dst_buf;
  bool m_src_buf_passed;
  bool m_dst_buf_passed;

  std::vector<int> m_peers;

  int get_num_peers() const {
    return m_peers.size();
  }

  void setup_rank_limits(const TensorType &src_tensor,
                         const TensorType &dst_tensor,
                         int *&rank_limits) {
    std::vector<int> host_buf;
    const int num_dims = src_tensor.get_num_dims();
    for (int i = 0; i < num_dims; ++i) {
      int dst_locale_dim = dst_tensor.get_locale_shape()[i];
      for (int j = 0; j < dst_locale_dim; ++j) {
        index_t next_rank_global_offset = dst_tensor.get_shape()[i];
        // Find the next split-root rank within dimension i.
        for (int next = j+1; next < dst_locale_dim; ++next) {
          if (dst_tensor.get_distribution().is_split_root(i, next)) {
            next_rank_global_offset = dst_tensor.get_dimension_rank_offset(i, next);
            break;
          }
        }
        index_t lim;
        // If the j-th limit is smaller than the base offset of the
        // source tensor, there is no overlap between the source and
        // destination tensors.
        if (next_rank_global_offset < src_tensor.get_global_index()[i]) {
          lim = 0;
        } else {
          lim = next_rank_global_offset - src_tensor.get_global_index()[i];
          // The offset of the next rank can be outside of this local tensor.
          lim = std::min(lim, src_tensor.get_local_shape()[i]);
        }
        host_buf.push_back(lim);
      }
    }
    // Optimization for a case where the dst tensor is evenly
    // partitioned.
#ifdef DISTCONV_OPTIMIZE_FIND_DESTINATION
    optimize_find_destination(src_tensor, dst_tensor, host_buf);
#endif
    DISTCONV_CUDA_MALLOC(&rank_limits,
                         sizeof(int) * host_buf.size());
    DISTCONV_CHECK_CUDA(cudaMemcpy(rank_limits,
                                   host_buf.data(),
                                   sizeof(int) * host_buf.size(),
                                   cudaMemcpyHostToDevice));
  }

  void optimize_find_destination(const TensorType &src_tensor,
                                 const TensorType &dst_tensor,
                                 std::vector<int> &rank_limits) {
    const int num_dims = src_tensor.get_num_dims();
    int rank_limits_idx = 0;
    for (int i = 0; i < num_dims; ++i) {
      int dst_locale_dim = dst_tensor.get_locale_shape()[i];
      // Check if the dimension is evenly partitioned
      auto reference_partition_size = dst_tensor.get_local_shape()[i];
      auto evenly_partitioned = true;
      for (int j = 0; j < dst_locale_dim; ++j) {
        auto partition_size = dst_tensor.get_remote_dimension(i, j);
        if (partition_size != reference_partition_size) {
          evenly_partitioned = false;
          break;
        }
      }
      // At least 3 entries are needed and this is probably only
      // meaningful for a relatively large dimension. Skip dimension
      // if it's shorter.
      const int opt_dim_threshold = 16;
      if (evenly_partitioned && dst_locale_dim >= opt_dim_threshold) {
        // This dimension is evenly partitioned and has enough space to
        // store necessary indices in rank_limits.
        // Mark that this is a special optimized case with -1
        rank_limits.at(rank_limits_idx) = -1;
        // Store the global offset of the source tensor. Note
        // rank_limits is an int array, but the global index is returned
        // as a size_t array. This should be fine as int should be
        // sufficient.
        rank_limits.at(rank_limits_idx+1) = src_tensor.get_global_index()[i];
        // Store the partition size of the destination tensor so that
        // the offset at the destination tensor can be calculated.
        rank_limits.at(rank_limits_idx+2) = reference_partition_size;
      }
      rank_limits_idx += dst_locale_dim;
    }
  }

  void setup_displs(const TensorType &src_tensor,
                    const TensorType &dst_tensor) {
    int num_ranks = m_loc.get_size();

    m_send_counts = new int[num_ranks];
    m_recv_counts = new int[num_ranks];
    m_send_displs_h = new int[num_ranks];
    m_recv_displs_h = new int[num_ranks];
    DISTCONV_CUDA_MALLOC(
        &m_send_displs_d, sizeof(int) * num_ranks);
    DISTCONV_CUDA_MALLOC(
        &m_recv_displs_d, sizeof(int) * num_ranks);

    const Region src_local_region(src_tensor.get_global_index(),
                                  m_src_local_shape);
    const Region dst_local_region(dst_tensor.get_global_index(),
                                  m_dst_local_shape);
    const auto &loc_shape_dst = m_dst_locale_shape;
    const auto &loc_shape_src = m_src_locale_shape;
    int cur_send_displs = 0;
    int cur_recv_displs = 0;

    bool src_split_root = src_tensor.is_split_root();
    bool dst_split_root = dst_tensor.is_split_root();

    util::MPIPrintStreamDebug()
        << "src_local_region: " << src_local_region
        << ", dst_local_region: " << dst_local_region
        << ", src_split_root: " << src_split_root
        << ", dst_split_root: " << dst_split_root;

    // transfers only between split root ranks
    for (int pid = 0; pid < num_ranks; ++pid) {
      m_send_displs_h[pid] = cur_send_displs;
      m_recv_displs_h[pid] = cur_recv_displs;
      // send_counts & send_displs
      const auto &dst_pid_idx = loc_shape_dst.get_index(pid);
      if (src_split_root &&
          dst_tensor.get_distribution().is_split_root(dst_pid_idx)) {
        Region dst_remote_region(
            dst_tensor.get_remote_index(dst_pid_idx),
            dst_tensor.get_remote_shape(dst_pid_idx));
        auto &&send_intersection =
            src_local_region.intersect(dst_remote_region);
        m_send_counts[pid] = send_intersection.get_size();
        util::MPIPrintStreamDebug()
            << "send_intersection for " << pid << ": "
            << send_intersection
            << ", dst_remote_region: " << dst_remote_region;
      } else {
        // do not send anything if the destination is not a split root
        util::MPIPrintStreamDebug() << "destination "
                                    << pid << " is not a split root";
        m_send_counts[pid] = 0;
      }
      cur_send_displs += m_send_counts[pid];
      // recv_counts & recv_displs
      const auto src_pid_idx = loc_shape_src.get_index(pid);
      if (dst_split_root &&
          src_tensor.get_distribution().is_split_root(src_pid_idx)) {
        Region src_remote_region(
            src_tensor.get_remote_index(src_pid_idx),
            src_tensor.get_remote_shape(src_pid_idx));
        auto &&recv_intersection =
            dst_local_region.intersect(src_remote_region);
        m_recv_counts[pid] = recv_intersection.get_size();
      } else {
        // similarly, if the remote source is not a split root, do not
        // receive anything from it
        util::MPIPrintStreamDebug() << "source is not a split root";
        m_recv_counts[pid] = 0;
      }
      cur_recv_displs += m_recv_counts[pid];

      util::MPIPrintStreamDebug()
          << "send displs for rank " << pid << ": " << m_send_displs_h[pid]
          << ", recv displs: " << m_recv_displs_h[pid]
          << ", send count: " << m_send_counts[pid]
          << ", recv count: " << m_recv_counts[pid];
    }
    DISTCONV_CHECK_CUDA(cudaMemcpy(m_send_displs_d,
                                   m_send_displs_h,
                                   sizeof(int) * num_ranks,
                                   cudaMemcpyHostToDevice));
    DISTCONV_CHECK_CUDA(cudaMemcpy(m_recv_displs_d,
                                   m_recv_displs_h,
                                   sizeof(int) * num_ranks,
                                   cudaMemcpyHostToDevice));
  }

  void shuffle(const DataType *src, DataType *dst,
               cudaStream_t stream, bool is_forward);

  virtual DataType *get_src_buf(bool is_forward, cudaStream_t s) {
    if (is_forward && m_src_buf_passed) {
      return m_src_buf;
    } else if (!is_forward && m_dst_buf_passed) {
      return m_dst_buf;
    } else {
      size_t buffer_size = get_src_local_shape(is_forward).get_size() *
          sizeof(DataType);
      DataType *buf = buffer_size == 0 ? nullptr :
          static_cast<DataType*>(
              distconv::internal::RuntimeCUDA::get_device_memory_pool().get(
                  buffer_size, s));
      return buf;
    }
  }

  virtual DataType *get_dst_buf(bool is_forward, cudaStream_t s) {
    if (is_forward && m_dst_buf_passed) {
      return m_dst_buf;
    } else if (!is_forward && m_src_buf_passed) {
      return m_src_buf;
    } else {
      size_t buffer_size = get_dst_local_shape(is_forward).get_size() *
          sizeof(DataType);
      DataType *buf = buffer_size == 0 ? nullptr :
          static_cast<DataType*>(
              distconv::internal::RuntimeCUDA::get_device_memory_pool().get(
                  buffer_size, s));
      return buf;
    }
  }

  virtual void transfer(
      const DataType *send_buf, size_t send_buffer_size,
      DataType *recv_buf, size_t recv_buffer_size,
      bool is_forward, cudaStream_t stream) {
#ifdef DISTCONV_SHFL_USE_CUDA_AWARE
    DISTCONV_CHECK_CUDA(cudaStreamSynchronize(stream));
    MPI_Alltoallv(send_buf,
                  get_send_counts(is_forward),
                  get_send_displs_h(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  recv_buf,
                  get_recv_counts(is_forward),
                  get_recv_displs_h(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  m_loc.get_comm());
#else
    // manually copying back to host
    DataType *send_buf_h = send_buffer_size == 0 ? nullptr :
        static_cast<DataType*>(
            tensor::internal::RuntimeCUDA::get_pinned_memory_pool().get(
                send_buffer_size));
    DataType *recv_buf_h = recv_buffer_size == 0 ? nullptr :
        static_cast<DataType*>(
            internal::RuntimeCUDA::get_pinned_memory_pool().get(
                recv_buffer_size));

    if (send_buffer_size > 0) {
      DISTCONV_CHECK_CUDA(cudaMemcpy(
          send_buf_h, send_buf,
          send_buffer_size, cudaMemcpyDeviceToHost));
    }

    MPI_Alltoallv(send_buf_h, get_send_counts(is_forward),
                  get_send_displs_h(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  recv_buf_h, get_recv_counts(is_forward),
                  get_recv_displs_h(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  m_loc.get_comm());

    if (recv_buffer_size > 0) {
      DISTCONV_CHECK_CUDA(cudaMemcpy(
          recv_buf, recv_buf_h,
          recv_buffer_size, cudaMemcpyHostToDevice));
    }

    if (send_buf_h != nullptr) {
      tensor::internal::RuntimeCUDA::get_pinned_memory_pool().release(send_buf_h);
    }
    if (recv_buf_h != nullptr) {
      tensor::internal::RuntimeCUDA::get_pinned_memory_pool().release(recv_buf_h);
    }
#endif
    util::MPIPrintStreamDebug() << "Transfer done\n";
  }

  virtual void release_buf(DataType *buf) {
    if (buf != nullptr && buf != m_src_buf && buf != m_dst_buf) {
      distconv::internal::RuntimeCUDA::get_device_memory_pool().release(buf);
    }
  }

  const Shape &get_src_local_shape(bool is_forward) const {
    if (is_forward) {
      return m_src_local_shape;
    } else {
      return m_dst_local_shape;
    }
  }

  const Shape &get_dst_local_shape(bool is_forward) const {
    if (is_forward) {
      return m_dst_local_shape;
    } else {
      return m_src_local_shape;
    }
  }

  const IndexVector &get_src_strides(bool is_forward) const {
    if (is_forward) {
      return m_src_strides;
    } else {
      return m_dst_strides;
    }
  }

  const IndexVector &get_dst_strides(bool is_forward) const {
    if (is_forward) {
      return m_dst_strides;
    } else {
      return m_src_strides;
    }
  }

  const Shape &get_src_locale_shape(bool is_forward) const {
    if (is_forward) {
      return m_src_locale_shape;
    } else {
      return m_dst_locale_shape;
    }
  }

  const Shape &get_dst_locale_shape(bool is_forward) const {
    if (is_forward) {
      return m_dst_locale_shape;
    } else {
      return m_src_locale_shape;
    }
  }

  const IntVector &get_src_overlap(bool is_forward) const {
    if (is_forward) {
      return m_src_overlap;
    } else {
      return m_dst_overlap;
    }
  }

  const IntVector &get_dst_overlap(bool is_forward) const {
    if (is_forward) {
      return m_dst_overlap;
    } else {
      return m_src_overlap;
    }
  }

  bool is_src_split_root(bool is_forward) const {
    return is_forward ? m_src_split_root : m_dst_split_root;
  }
  bool is_dst_split_root(bool is_forward) const {
    return is_forward ? m_dst_split_root : m_src_split_root;
  }

  const int *get_rank_limits_fwd(bool is_forward) const {
    return is_forward ? m_rank_limits_fwd : m_rank_limits_bwd;
  }
  const int *get_rank_limits_bwd(bool is_forward) const {
    return is_forward ? m_rank_limits_bwd : m_rank_limits_fwd;
  }

  const int *get_send_counts(bool is_forward) const {
    return is_forward ? m_send_counts : m_recv_counts;
  }
  const int *get_recv_counts(bool is_forward) const {
    return is_forward ? m_recv_counts : m_send_counts;
  }

  const int *get_send_displs_h(bool is_forward) const {
    return is_forward ? m_send_displs_h : m_recv_displs_h;
  }
  const int *get_recv_displs_h(bool is_forward) const {
    return is_forward ? m_recv_displs_h : m_send_displs_h;
  }

  const int *get_send_displs_d(bool is_forward) const {
    return is_forward ? m_send_displs_d : m_recv_displs_d;
  }
  const int *get_recv_displs_d(bool is_forward) const {
    return is_forward ? m_recv_displs_d : m_send_displs_d;
  }
};

} // namespace tensor
} // namespace distconv
