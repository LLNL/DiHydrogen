#pragma once

#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_mpi.hpp"
#include "distconv/util/util_gpu.hpp" // for profiler marking
#include "distconv/util/util_mpi.hpp"

#include <algorithm>
#include <cstring>

#define CALC_OFFSET4(i0, i1, i2, i3, strides)                           \
  ((i0) * strides[0] + (i1) * strides[1] + (i2) * strides[2] +          \
   (i3) * strides[3])

#define CALC_OFFSET5(i0, i1, i2, i3, i4, strides)                       \
  ((i0) * strides[0] + (i1) * strides[1] + (i2) * strides[2] +          \
   (i3) * strides[3] + (i4) * strides[4])

namespace distconv {
namespace tensor {
namespace internal {

template <typename DataType, typename Allocator>
class TensorMPIShuffleHelper {
  using TensorType = Tensor<DataType, LocaleMPI, Allocator>;
  using StreamType = typename Stream<Allocator>::type;
 public:
  TensorMPIShuffleHelper(const TensorType &src_tensor,
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
      m_src_buf(src_buf), m_dst_buf(dst_buf),
      m_src_buf_passed(src_buf != nullptr),
      m_dst_buf_passed(dst_buf != nullptr) {
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

  virtual ~TensorMPIShuffleHelper() = default;

  static size_t get_buf_size(const TensorType &tensor) {
    return get_buf_size(tensor.get_local_shape());
  }

  static size_t get_buf_size(const Shape &tensor_local_shape) {
    return tensor_local_shape.get_size() * sizeof(DataType);
  }

  const Shape m_src_local_shape;
  const Shape m_dst_local_shape;
  const IndexVector m_src_strides;
  const IndexVector m_dst_strides;
  const Shape m_src_locale_shape;
  const Shape m_dst_locale_shape;
  const IntVector m_src_overlap;
  const IntVector m_dst_overlap;
  const LocaleMPI m_loc;
  const bool m_src_split_root;
  const bool m_dst_split_root;

  // Offsets in src tensor for each dst locale. Used in
  // packing. Linearized to a 1D array.
  std::vector<int> m_rank_limits_fwd;
  // Offsets in dst tensor for each src locale. Used in
  // packing. Linearized to a 1D array.
  std::vector<int> m_rank_limits_bwd;
  std::vector<int> m_send_counts;
  std::vector<int> m_recv_counts;
  std::vector<int> m_send_displs;
  std::vector<int> m_recv_displs;

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
                         std::vector<int> &rank_limits) {
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
    rank_limits = host_buf;
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

    m_send_counts = std::vector<int>(num_ranks);
    m_recv_counts = std::vector<int>(num_ranks);
    m_send_displs = std::vector<int>(num_ranks);
    m_recv_displs = std::vector<int>(num_ranks);

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
      m_send_displs[pid] = cur_send_displs;
      m_recv_displs[pid] = cur_recv_displs;
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
          << "send displs for rank " << pid << ": " << m_send_displs[pid]
          << ", recv displs: " << m_recv_displs[pid]
          << ", send count: " << m_send_counts[pid]
          << ", recv count: " << m_recv_counts[pid];
    }
  }

  int get_num_dims() const {
    return m_src_locale_shape.num_dims();
  }

  template <typename BufAlloc, typename BufDel>
  std::shared_ptr<DataType> get_src_buf(
      bool is_forward, StreamType stream,
      BufAlloc alloc, BufDel del) {
    auto non_delete = [] (DataType *p) {};
    if (is_forward && m_src_buf_passed) {
      return std::shared_ptr<DataType>(m_src_buf, non_delete);
    } else if (!is_forward && m_dst_buf_passed) {
      return std::shared_ptr<DataType>(m_dst_buf, non_delete);
    } else {
      size_t buffer_count = get_src_local_shape(is_forward).size();
      DataType *buf = buffer_count == 0 ? nullptr :
          alloc(buffer_count, stream);
      return std::shared_ptr<DataType>(buf, del);
    }
  }

  template <typename BufAlloc, typename BufDel>
  std::shared_ptr<DataType> get_dst_buf(
      bool is_forward, StreamType stream,
      BufAlloc alloc, BufDel del) {
    auto non_delete = [] (DataType *p) {};
    if (is_forward && m_dst_buf_passed) {
      return std::shared_ptr<DataType>(m_dst_buf, non_delete);
    } else if (!is_forward && m_src_buf_passed) {
      return std::shared_ptr<DataType>(m_src_buf, non_delete);
    } else {
      size_t buffer_count = get_dst_local_shape(is_forward).size();
      DataType *buf = buffer_count == 0 ? nullptr :
          alloc(buffer_count, stream);
      return std::shared_ptr<DataType>(buf, del);
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
    return is_forward ? m_rank_limits_fwd.data() : m_rank_limits_bwd.data();
  }
  const int *get_rank_limits_bwd(bool is_forward) const {
    return is_forward ? m_rank_limits_bwd.data() : m_rank_limits_fwd.data();
  }

  const int *get_send_counts(bool is_forward) const {
    return is_forward ? m_send_counts.data() : m_recv_counts.data();
  }
  const int *get_recv_counts(bool is_forward) const {
    return is_forward ? m_recv_counts.data() : m_send_counts.data();
  }

  const int *get_send_displs(bool is_forward) const {
    return is_forward ? m_send_displs.data() : m_recv_displs.data();
  }
  const int *get_recv_displs(bool is_forward) const {
    return is_forward ? m_recv_displs.data() : m_send_displs.data();
  }

};
} // namespace internal

template <typename DataType, typename Allocator>
class TensorMPIShuffler;

// Partial specialization for BaseAllocator
template <typename DataType>
class TensorMPIShuffler<DataType, BaseAllocator> {
 protected:
  using Allocator = BaseAllocator;
  using TensorType = Tensor<DataType, LocaleMPI, Allocator>;
  using StreamType = typename Stream<Allocator>::type;
  static constexpr StreamType default_stream = Stream<Allocator>::default_value;
 public:

  TensorMPIShuffler(const TensorType &src_tensor,
                    const TensorType &dst_tensor,
                    DataType *src_buf=nullptr,
                    DataType *dst_buf=nullptr):
      m_helper(src_tensor, dst_tensor, src_buf, dst_buf) {
    assert0(src_tensor.get_overlap().reduce_sum());
    m_fwd_sample_to_spatial = is_sample_to_spatial(src_tensor, dst_tensor);
    m_bwd_sample_to_spatial = is_sample_to_spatial(dst_tensor, src_tensor);
  }

  virtual ~TensorMPIShuffler() = default;

  void shuffle_forward(
      const DataType *src, DataType *dst,
      StreamType stream=default_stream) {
    shuffle(src, dst, stream, true);
  }

  void shuffle_backward(
      const DataType *src, DataType *dst,
      StreamType stream=default_stream) {
    shuffle(src, dst, stream, false);
  }

  static size_t get_buf_size(const TensorType &tensor) {
    return internal::TensorMPIShuffleHelper<
      DataType, Allocator>::get_buf_size(tensor);
  }

 protected:
  internal::TensorMPIShuffleHelper<DataType, Allocator> m_helper;
  bool m_fwd_sample_to_spatial;
  bool m_bwd_sample_to_spatial;

  bool is_sample_to_spatial(const TensorType &src,
                            const TensorType &dst) {
    const int nd = src.get_num_dims();

    // Only 4D and 5D tensors
    if (nd != 4 && nd != 5) return false;

    // The source tensor must not have splitting other than the sample
    // dimension.
    for (int i = 0; i < nd - 1; ++i) {
      if (src.get_split_shape()[i] != 1) return false;
    }

    // No sharing on the sample dimension
    if (src.get_split_shape()[-1] != src.get_locale_shape()[-1]) return false;
    if (dst.get_split_shape()[-1] != dst.get_locale_shape()[-1]) return false;

    // Unpack assumes only receving from a single source, which means
    // the sample dimension must be equal size
    if (src.get_locale_shape()[-1] != dst.get_locale_shape()[-1]) return false;

    // Assumes the dest tensor is evenly partitioned except for the
    // sample dimension, whose partitioning is not changed.
    for (int i = 0; i < nd - 1; ++i) {
      if (dst.get_local_shape()[i] * dst.get_locale_shape()[i] !=
          dst.get_shape()[i]) return false;
    }

    return true;
  }

  bool get_sample_to_spatial(bool is_forward) {
    return is_forward ? m_fwd_sample_to_spatial :
        m_bwd_sample_to_spatial;
  }

  virtual void shuffle(const DataType *src, DataType *dst,
                       StreamType stream, bool is_forward) {
    // Poiners can be null if they are empty, which can happen in MPI
    // local tensors
    //assert_always(src != nullptr);
    //assert_always(dst != nullptr);

    // No overlap supported for source tensors
    assert0(m_helper.get_src_overlap(is_forward).reduce_sum());
    // The unpack-opt supports overlaped tensors
    if (!get_sample_to_spatial(is_forward)) {
      assert_always(m_helper.get_dst_overlap(is_forward).reduce_sum() == 0);
    }

    const int *rank_limits_fwd = m_helper.get_rank_limits_fwd(is_forward);
    const int *rank_limits_bwd = m_helper.get_rank_limits_bwd(is_forward);
    const int *send_displs = m_helper.get_send_displs(is_forward);
    const int *recv_displs = m_helper.get_recv_displs(is_forward);

    auto send_buf = m_helper.get_src_buf(
        is_forward, stream,
        [](size_t c, StreamType s) { return new DataType[c]; },
        [](DataType *p) { delete[] p; });
    auto recv_buf = m_helper.get_dst_buf(
        is_forward, stream,
        [](size_t c, StreamType s) { return new DataType[c]; },
        [](DataType *p) { delete[] p; });

    int nd = m_helper.get_num_dims();

    util::profile_push("pack");

    if (!getenv("SKIP_PACK")) {
      if (m_helper.is_src_split_root(is_forward)) {
        if (get_sample_to_spatial(is_forward) &&
            (nd == 4 || nd == 5)) {
          util::MPIPrintStreamDebug() << "Sample-to-spatial packing";
          util::profile_push("pack-opt");
          if (nd == 4) {
            pack_sample_to_spatial4(
                src, m_helper.get_src_local_shape(is_forward),
                m_helper.get_dst_local_shape(is_forward),
                m_helper.get_dst_locale_shape(is_forward),
                send_buf.get());
          } else if (nd == 5) {
            pack_sample_to_spatial5(
                src, m_helper.get_src_local_shape(is_forward),
                m_helper.get_dst_local_shape(is_forward),
                m_helper.get_dst_locale_shape(is_forward),
                send_buf.get());
          }
          util::profile_pop();
        } else {
            util::profile_push("pack-default");
            util::MPIRootPrintStreamWarning()
                << "Packing does not use the optimized implementation";
            pack(src,
                 m_helper.get_src_local_shape(is_forward),
                 m_helper.get_src_strides(is_forward),
                 m_helper.get_dst_locale_shape(is_forward),
                 rank_limits_fwd,
                 send_buf.get(),
                 send_displs);
            util::profile_pop();
        }
      }
    }

    util::profile_pop(); // pack

    util::profile_push("transfer");
    if (!getenv("SKIP_TRANSFER")) {
      transfer(send_buf, recv_buf, is_forward);
    }
    util::profile_pop();

    util::profile_push("unpack");
    // unpack
    if (!getenv("SKIP_UNPACK")) {
      if (m_helper.is_dst_split_root(is_forward)) {
        if (get_sample_to_spatial(is_forward)) {
            util::profile_push("unpack-opt");
            util::MPIPrintStreamDebug() << "Sample-to-spatial unpacking";
            if (nd == 4)
            {
                unpack_sample_to_spatial_halo4(
                    dst,
                    m_helper.get_dst_local_shape(is_forward),
                    m_helper.get_dst_strides(is_forward),
                    recv_buf.get(),
                    m_helper.get_dst_overlap(is_forward));
            }
            else
            {
                unpack_sample_to_spatial_halo5(
                    dst,
                    m_helper.get_dst_local_shape(is_forward),
                    m_helper.get_dst_strides(is_forward),
                    recv_buf.get(),
                    m_helper.get_dst_overlap(is_forward));
            }
            util::profile_pop();
        } else {
            util::profile_push("unpack-default");
            unpack(dst,
                   m_helper.get_dst_local_shape(is_forward),
                   m_helper.get_dst_strides(is_forward),
                   m_helper.get_src_locale_shape(is_forward),
                   rank_limits_bwd,
                   recv_buf.get(),
                   recv_displs);
            util::profile_pop();
        }
      }
    }
    util::profile_pop();
  }

  virtual void transfer(const std::shared_ptr<DataType> &send_buf,
                        std::shared_ptr<DataType> &recv_buf,
                        bool is_forward) {
    MPI_Alltoallv(send_buf.get(),
                  m_helper.get_send_counts(is_forward),
                  m_helper.get_send_displs(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  recv_buf.get(),
                  m_helper.get_recv_counts(is_forward),
                  m_helper.get_recv_displs(is_forward),
                  util::get_mpi_data_type<DataType>(),
                  m_helper.m_loc.get_comm());
    util::MPIPrintStreamDebug() << "Transfer done";
  }

#if 0
  virtual void transfer_sample_to_spatial(
      const std::shared_ptr<DataType> &send_buf,
      std::shared_ptr<DataType> &recv_buf,
      bool is_forward) {
    MPI_Scatter(send_buf.get(),
                m_helper.get_send_counts(is_forward)[0],
                util::get_mpi_data_type<DataType>(),
                recv_buf.get(),
                m_helper.get_recv_counts(is_forward)[0],
                util::get_mpi_data_type<DataType>(),
                0,
                m_helper.m_loc.get_comm());
                //m_dst_spatial_locale.get_comm());
    util::MPIPrintStreamDebug() << "Transfer done";
  }
#endif

  void find_destination(const IndexVector &src_local_idx,
                        const Shape &src_local_shape,
                        const Shape &dst_locale_shape,
                        const int * __restrict__ rank_limits,
                        int &dst_rank, size_t &dst_offset) {
    dst_rank = 0;
    dst_offset = 0;
    int rank_dim_offset = 1;
    size_t local_linear_offset = 1;
    int rank_limits_idx = 0;
    const int ND = src_local_idx.length();
    for (int i = 0; i < ND; ++i) {
      // Locate the i-th dim index of the destination rank
      int dst_rank_idx;
      int dst_buffer_offset;
      int dst_buffer_dim;
#ifdef DISTCONV_OPTIMIZE_FIND_DESTINATION
      if (rank_limits[rank_limits_idx] == -1) {
        auto src_global_index = src_local_idx[i] + rank_limits[rank_limits_idx+1];
        int dst_local_dim = rank_limits[rank_limits_idx+2];
        dst_rank_idx = src_global_index / dst_local_dim;
        dst_buffer_offset = std::min(src_global_index % dst_local_dim,
                                     src_local_idx[i]);
        dst_buffer_dim = std::min((int)(src_local_shape[i] -
                                        (src_local_idx[i] - dst_buffer_offset)),
                                  dst_local_dim);
      } else {
#endif
        // The if-condition below always holds for some j, so dst_rank_idx
        // is always assigned some value. Initialize just to suppress
        // compiler warnings.
        dst_rank_idx = 0;
        for (int j = 0; j < (int)dst_locale_shape[i]; ++j) {
          if (static_cast<int>(src_local_idx[i]) <
              rank_limits[rank_limits_idx+j]) {
            dst_rank_idx = j;
            break;
          }
        }
        int dst_rank_base = dst_rank_idx == 0 ? 0 :
            rank_limits[rank_limits_idx+dst_rank_idx-1];
        dst_buffer_offset = src_local_idx[i] - dst_rank_base;
        dst_buffer_dim = rank_limits[rank_limits_idx+dst_rank_idx]
            - dst_rank_base;
#ifdef DISTCONV_OPTIMIZE_FIND_DESTINATION
      }
#endif

      dst_rank += dst_rank_idx * rank_dim_offset;
      rank_dim_offset *= dst_locale_shape[i];

      dst_offset += dst_buffer_offset * local_linear_offset;
      local_linear_offset *= dst_buffer_dim;

      rank_limits_idx += dst_locale_shape[i];
    }
  }

  // NOTE: packed tensor is assumed
  void pack(const DataType *src, const Shape &src_local_shape,
            const IndexVector &src_strides, const Shape &dst_locale_shape,
            const int *rank_limits, DataType *buf,
            const int *displs) {
    if (src_local_shape.size() == 0) return;

    const size_t size = src_local_shape.size();
    const size_t gid = 0;
    const size_t num_threads = 1;

    for (size_t offset = gid; offset < size; offset += num_threads) {
      DataType v = src[offset];
      const auto idx = src_local_shape.get_index(offset);
      int rank;
      size_t packed_buf_offset;
      find_destination(idx, src_local_shape, dst_locale_shape,
                       rank_limits, rank, packed_buf_offset);
      buf[displs[rank] + packed_buf_offset] = v;
    }
  }

  void pack_sample_to_spatial4(
      const DataType *src, const Shape &src_local_shape,
      const Shape &dst_local_shape,
      const Shape &dst_locale_shape,
      DataType *buf) {
    constexpr int ND = 4;
    if (src_local_shape.size() == 0) return;

    auto dst_local_size = dst_local_shape.size();
    auto dst_offset = 0;
    auto num_dst_ranks = dst_locale_shape.size() / dst_locale_shape[-1];
    index_t *dst_offsets = new index_t[num_dst_ranks];
    int dst_offsets_idx = 0;
    for (int p2 = 0; p2 < (int)dst_locale_shape[2]; ++p2) {
      for (int p1 = 0; p1 < (int)dst_locale_shape[1]; ++p1) {
        for (int p0 = 0; p0 < (int)dst_locale_shape[0]; ++p0) {
          dst_offsets[dst_offsets_idx++] = dst_offset;
          dst_offset += dst_local_size;
        }
      }
    }

    Array<ND> src_local_strides;
    Array<ND> dst_local_strides;
    Array<ND> dst_locale_strides;
    index_t src_stride = 1;
    index_t dst_stride = 1;
    int dst_locale_stride = 1;
    for (int i = 0; i < ND; ++i) {
      src_local_strides[i] = src_stride;
      dst_local_strides[i] = dst_stride;
      dst_locale_strides[i] = dst_locale_stride;
      src_stride *= src_local_shape[i];
      dst_stride *= dst_local_shape[i];
      dst_locale_stride *= dst_locale_shape[i];
    }

    const int linear_len = dst_local_shape[0];
    constexpr int p3 = 0;
#pragma omp parallel for collapse(5)
    for (int i3 = 0; i3 < (int)dst_local_shape[3]; ++i3) {
      for (int p2 = 0; p2 < (int)dst_locale_shape[2]; ++p2) {
        for (int i2 = 0; i2 < (int)dst_local_shape[2]; ++i2) {
          for (int p1 = 0; p1 < (int)dst_locale_shape[1]; ++p1) {
            for (int i1 = 0; i1 < (int)dst_local_shape[1]; ++i1) {
              index_t src_offset =
                  CALC_OFFSET4(
                      0, i1 + dst_local_shape[1] * p1, i2 + dst_local_shape[2] * p2,
                      i3 + dst_local_shape[3] * p3, src_local_strides);
              index_t dst_offset_i12 = CALC_OFFSET4(0, i1, i2, i3, dst_local_strides);
              for (int p0 = 0; p0 < (int)dst_locale_shape[0]; ++p0) {
                int dst_rank_idx = CALC_OFFSET4(p0, p1, p2, p3, dst_locale_strides);
                std::memcpy(&buf[dst_offsets[dst_rank_idx]+dst_offset_i12],
                            &src[src_offset],
                            sizeof(DataType) * linear_len);
                src_offset += linear_len;
              }
            }
          }
        }
      }
    }
    delete[] dst_offsets;
  }

  void pack_sample_to_spatial5(
      const DataType *src, const Shape &src_local_shape,
      const Shape &dst_local_shape,
      const Shape &dst_locale_shape,
      DataType *buf) {
    constexpr int ND = 5;
    if (src_local_shape.size() == 0) return;

    auto num_dst_ranks = dst_locale_shape.size() / dst_locale_shape[-1];
    index_t *dst_offsets = new index_t[num_dst_ranks];
    {
      auto dst_local_size = dst_local_shape.size();
      auto dst_offset = 0;
      int dst_offsets_idx = 0;
      for (int p3 = 0; p3 < (int)dst_locale_shape[3]; ++p3) {
        for (int p2 = 0; p2 < (int)dst_locale_shape[2]; ++p2) {
          for (int p1 = 0; p1 < (int)dst_locale_shape[1]; ++p1) {
            for (int p0 = 0; p0 < (int)dst_locale_shape[0]; ++p0) {
              dst_offsets[dst_offsets_idx++] = dst_offset;
              dst_offset += dst_local_size;
            }
          }
        }
      }
    }

    Array<ND> src_local_strides;
    Array<ND> dst_local_strides;
    Array<ND> dst_locale_strides;
    index_t src_stride = 1;
    index_t dst_stride = 1;
    int dst_locale_stride = 1;
    for (int i = 0; i < ND; ++i) {
      src_local_strides[i] = src_stride;
      dst_local_strides[i] = dst_stride;
      dst_locale_strides[i] = dst_locale_stride;
      src_stride *= src_local_shape[i];
      dst_stride *= dst_local_shape[i];
      dst_locale_stride *= dst_locale_shape[i];
    }

    const int linear_len = dst_local_shape[0];
    constexpr int p4 = 0;
#pragma omp parallel for collapse(5)
    for (int i4 = 0; i4 < (int)dst_local_shape[4]; ++i4) {
      for (int p3 = 0; p3 < (int)dst_locale_shape[3]; ++p3) {
        for (int i3 = 0; i3 < (int)dst_local_shape[3]; ++i3) {
          for (int p2 = 0; p2 < (int)dst_locale_shape[2]; ++p2) {
            for (int i2 = 0; i2 < (int)dst_local_shape[2]; ++i2) {
              for (int p1 = 0; p1 < (int)dst_locale_shape[1]; ++p1) {
                for (int i1 = 0; i1 < (int)dst_local_shape[1]; ++i1) {
                  for (int p0 = 0; p0 < (int)dst_locale_shape[0]; ++p0) {
                    index_t src_offset =
                        CALC_OFFSET5(
                            0  + dst_local_shape[0] * p0,
                            i1 + dst_local_shape[1] * p1,
                            i2 + dst_local_shape[2] * p2,
                            i3 + dst_local_shape[3] * p3,
                            i4 + dst_local_shape[4] * p4,
                            src_local_strides);
                    index_t dst_offset =
                        CALC_OFFSET5(0, i1, i2, i3, i4, dst_local_strides);
                    int dst_rank_idx =
                        CALC_OFFSET5(p0, p1, p2, p3, p4, dst_locale_strides);
                    std::memcpy(&buf[dst_offsets[dst_rank_idx]+dst_offset],
                                &src[src_offset],
                                sizeof(DataType) * linear_len);
                  }
                }
              }
            }
          }
        }
      }
    }
    delete[] dst_offsets;
  }

  void unpack(DataType *dst,
              const Shape &dst_local_shape,
              const IndexVector &dst_strides,
              const Shape &src_locale_shape,
              const int *rank_limits,
              const DataType *buf,
              const int *displs) {
    if (dst_local_shape.size() == 0) return;

    const size_t size = dst_local_shape.size();
    const size_t gid = 0;
    const size_t num_threads = 1;

    for (size_t offset = gid; offset < size; offset += num_threads) {
      const auto idx = dst_local_shape.get_index(offset);
      int rank;
      size_t packed_buf_offset;
      find_destination(idx, dst_local_shape, src_locale_shape,
                       rank_limits, rank, packed_buf_offset);
      dst[offset] = buf[displs[rank] + packed_buf_offset];
    }
  }

  void unpack_sample_to_spatial(DataType *dst,
                                const Shape &dst_local_shape,
                                const DataType *buf) {
    if (dst_local_shape.size() == 0) return;
    std::memcpy(dst, buf, dst_local_shape.size() * sizeof(DataType));
  }

  void unpack_sample_to_spatial_halo4(DataType *dst,
                                      const Shape &dst_local_shape,
                                      const IndexVector &dst_strides,
                                      const DataType *buf,
                                      const IntVector &dst_overlap) {
    constexpr int ND = 4;
    if (dst_local_shape.size() == 0) return;

    // packed strides
    Array<ND> buf_strides;
    index_t buf_stride = 1;
    for (int i = 0; i < ND; ++i) {
      buf_strides[i] = buf_stride;
      buf_stride *= dst_local_shape[i];
    }

    const int linear_len = dst_local_shape[0];
#pragma omp parallel for collapse(3)
    for (int i3 = 0; i3 < (int)dst_local_shape[3]; ++i3) {
      for (int i2 = 0; i2 < (int)dst_local_shape[2]; ++i2) {
        for (int i1 = 0; i1 < (int)dst_local_shape[1]; ++i1) {
          int i0 = 0;
          index_t dst_offset =
              CALC_OFFSET4(i0, i1, i2, i3, dst_strides);
          index_t buf_offset =
              CALC_OFFSET4(i0, i1, i2, i3, buf_strides);
          std::memcpy(&dst[dst_offset], &buf[buf_offset],
                      sizeof(DataType) * linear_len);
        }
      }
    }
  }

  void unpack_sample_to_spatial_halo5(DataType *dst,
                                      const Shape &dst_local_shape,
                                      const IndexVector &dst_strides,
                                      const DataType *buf,
                                      const IntVector &dst_overlap) {
    constexpr int ND = 4;
    if (dst_local_shape.size() == 0) return;

    // packed strides
    Array<ND> buf_strides;
    index_t buf_stride = 1;
    for (int i = 0; i < ND; ++i) {
      buf_strides[i] = buf_stride;
      buf_stride *= dst_local_shape[i];
    }

    const int linear_len = dst_local_shape[0];
#pragma omp parallel for collapse(4)
    for (int i4 = 0; i4 < (int)dst_local_shape[4]; ++i4) {
      for (int i3 = 0; i3 < (int)dst_local_shape[3]; ++i3) {
        for (int i2 = 0; i2 < (int)dst_local_shape[2]; ++i2) {
          for (int i1 = 0; i1 < (int)dst_local_shape[1]; ++i1) {
            constexpr int i0 = 0;
            index_t dst_offset =
                CALC_OFFSET5(i0, i1, i2, i3, i4, dst_strides);
            index_t buf_offset =
                CALC_OFFSET5(i0, i1, i2, i3, i4, buf_strides);
            std::memcpy(&dst[dst_offset], &buf[buf_offset],
                        sizeof(DataType) * linear_len);
          }
        }
      }
    }
  }
};

} // namespace tensor
} // namespace distconv

#undef CALC_OFFSET4
#undef CALC_OFFSET5
