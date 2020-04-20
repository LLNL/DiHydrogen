#pragma once

#include <cassert>
#include <type_traits>
#include <vector>
#include <cstring>

#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_process.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_mpi.hpp"

#include "mpi.h"

namespace distconv {
namespace tensor {

class LocaleMPI {
 public:
  LocaleMPI(MPI_Comm comm=MPI_COMM_WORLD):
      m_comm(new MPI_Comm, LocaleMPI::delete_comm) {
    *m_comm = comm;
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_num_procs);
  }

  LocaleMPI(MPI_Comm comm, bool release_ownership):
      m_comm(new MPI_Comm, LocaleMPI::delete_comm) {
    if (!release_ownership) {
      MPI_Comm comm2;
      MPI_Comm_dup(comm, &comm2);
      comm = comm2;
    }
    *m_comm = comm;
    MPI_Comm_rank(comm, &m_rank);
    MPI_Comm_size(comm, &m_num_procs);
  }

  int get_rank() const {
    return m_rank;
  }

  int get_size() const {
    return m_num_procs;
  }

  IndexVector get_rank_idx(const Distribution &dist) const {
    IndexVector rank_idx(dist.num_dims(), 0);
    const auto &locale_shape = dist.get_locale_shape();
    int rank = m_rank;
    for (int i = 0; i < dist.num_dims(); ++i) {
      rank_idx[i] = rank % locale_shape[i];
      rank = rank / locale_shape[i];
    }
    return rank_idx;
  }

  IndexVector get_split_idx(const Distribution &dist) const {
    auto idx = get_rank_idx(dist);
    for (int i = 0; i < dist.num_dims(); ++i) {
      idx[i] /= dist.get_num_ranks_per_split(i);
    }
    return idx;
  }

  bool is_split_root(const Distribution &dist) const {
    auto idx = get_rank_idx(dist);
    for (int i = 0; i < dist.num_dims(); ++i) {
      if (idx[i] % dist.get_num_ranks_per_split(i)) {
        return false;
      }
    }
    return true;
  }

  MPI_Comm get_comm() const {
    return *m_comm;
  }

 protected:
  static void delete_comm(MPI_Comm *p) {
    if (*p != MPI_COMM_WORLD && *p != MPI_COMM_NULL) {
      MPI_Comm_free(p);
    }
  }
#if 0
  MPI_Comm m_comm;
#else
  std::shared_ptr<MPI_Comm> m_comm;
#endif
  int m_rank;
  int m_num_procs;
};

template <typename DataType, typename Allocator>
class TensorImplHelper;

template <typename DataType, typename Allocator>
class TensorImpl<Tensor<DataType, LocaleMPI, Allocator>> {
  using TensorType = Tensor<DataType, LocaleMPI, Allocator>;
  using HelperType = TensorImplHelper<DataType, Allocator>;
  friend HelperType;
 public:
  TensorImpl(): m_tensor(nullptr) {}

  TensorImpl(TensorType *tensor):
      m_tensor(tensor),
      m_proc_idx(tensor->get_num_dims(), 0),
      m_split_idx(tensor->get_num_dims(), 0),
      m_local_shape(tensor->get_num_dims(), 0),
      m_local_real_shape(tensor->get_num_dims(), 0),
      m_max_local_shape(tensor->get_num_dims(), 0),
      m_offset(tensor->get_num_dims(), 0) {
    // cyclic distribution not supported
    if (m_tensor) {
      ensure_no_cyclic_distribution(m_tensor->get_distribution());
      init_proc_grid();
      init_local_tensor();
      init_offsets();
    }
  }

  TensorImpl(const TensorImpl<TensorType> &x):
      m_tensor(x.m_tensor), m_proc_idx(x.m_proc_idx),
      m_split_idx(x.m_split_idx),
      m_local_shape(x.m_local_shape),
      m_local_real_shape(x.m_local_real_shape),
      m_max_local_shape(x.m_max_local_shape),
      m_offset(x.m_offset), m_offset_all(x.m_offset_all) {}

  TensorImpl(TensorType *tensor, const TensorImpl<TensorType> &x):
      m_tensor(tensor), m_proc_idx(x.m_proc_idx),
      m_split_idx(x.m_split_idx),
      m_local_shape(x.m_local_shape),
      m_local_real_shape(x.m_local_real_shape),
      m_max_local_shape(x.m_max_local_shape),
      m_offset(x.m_offset), m_offset_all(x.m_offset_all) {}

  TensorImpl<TensorType> &operator=(const TensorImpl<TensorType> &x) {
    m_tensor = x.m_tensor;
    m_proc_idx = x.m_proc_idx;
    m_split_idx = x.m_split_idx;
    m_local_shape = x.m_local_shape;
    m_local_real_shape = x.m_local_real_shape;
    m_max_local_shape = x.m_max_local_shape;
    m_offset = x.m_offset;
    m_offset_all = x.m_offset_all;
    return *this;
  }

  ~TensorImpl() = default;

  void set_shape(const Shape &shape) {
    m_tensor->m_shape = shape;
    init_local_tensor();
    init_offsets();
  }

  void set_distribution(const Distribution &dist) {
    m_tensor->get_distribution() = dist;
    init_proc_grid();
    init_local_tensor();
    init_offsets();
  }

  // No memory reallocation is done. Assumes global_dim not to exceed
  // the original value of the outermost dimension.
  void set_outermost_dimension(index_t global_dim) {
    index_t old_dim = m_tensor->get_shape()[-1];
    if (old_dim == global_dim) return;
    util::MPIPrintStreamDebug() << "Changing the outermost dimension from "
                                << old_dim << " to " << global_dim;
    m_tensor->m_shape[-1] = global_dim;
    m_tensor->m_requested_local_shape[-1] = 0;
    init_local_tensor();

    // m_offset and m_offset_all needs to be updated. The partitioning
    // is assumed to be done without using requested sizes.
    int num_procs = m_tensor->get_distribution().get_locale_shape()[-1];
    int num_splits = m_tensor->get_distribution().get_split_shape()[-1];
    index_t num_procs_per_split =
        m_tensor->get_distribution().get_num_ranks_per_split()[-1];
    index_t local_dim = global_dim / num_splits;
    int rem = global_dim % num_splits;
    auto &offsets = m_offset_all.back();
    index_t cur_offset = 0;
    for (int i = 0; i < num_procs; i += num_procs_per_split) {
      for (int j = 0; j < num_procs_per_split; ++j) {
        offsets[i+j] = cur_offset;
      }
      cur_offset += local_dim;
      if (i < rem)  ++cur_offset;
    }
    assert_always(offsets[num_procs - 1] + local_dim == global_dim);
    m_offset[-1] = offsets[m_proc_idx[-1]];
  }

  Shape get_local_shape(bool include_halo) const {
    return include_halo ? m_local_real_shape : m_local_shape;
  }

  Shape get_local_shape() const {
    return get_local_shape(false);
  }

  Shape get_max_local_shape() const {
    return m_max_local_shape;
  }

  Shape get_max_local_real_shape() const {
    auto s = get_max_local_shape();
    for (int i = 0; i < m_tensor->get_num_dims(); ++i) {
      s[i] += m_tensor->get_halo_width(i) * 2;
    }
    return s;
  }

  index_t get_local_size() const {
    return get_local_shape().get_size();
  }

  index_t get_local_real_size() const {
    return get_local_shape(true).get_size();
  }

  int allocate() {
    const auto &dist = m_tensor->get_distribution();
    const auto &locale_shape = dist.get_locale_shape();
    // MPI num procs must be equal to the locale shape size, except
    // for shared tensors
    util::MPIPrintStreamDebug()
        << "locale size: " << m_tensor->m_locale.get_size()
        << ", locale: " << locale_shape
        << ", shape size: " << locale_shape.size();
    if (dist.is_distributed()) {
      index_t num_procs = m_tensor->m_locale.get_size();
      assert_always(num_procs == locale_shape.size());
    }

    auto num_local_elements = get_local_real_size();
    util::MPIPrintStreamDebug()
        << "num_local_elements: " << num_local_elements;
    if (num_local_elements > 0) {
      m_tensor->m_data.allocate(num_local_elements * sizeof(DataType),
                                get_local_real_shape()[0] * sizeof(DataType));
    } else {
      util::MPIPrintStreamInfo() << "Ignoring allocation of an empty tensor";
    }
    return 0;
  }

  void nullify() {
    m_tensor->m_data.nullify();
  }

  Shape get_local_real_shape() const {
    return get_local_shape(true);
  }

  index_t get_global_index(int dim, index_t local_idx) const {
    return m_offset[dim] + local_idx;
  }

  index_t get_local_index(int dim, index_t global_idx) const {
    return global_idx - m_offset[dim];
  }

  index_t get_local_offset(const IndexVector &idx,
                           bool idx_include_halo) const {
    auto real_idx = idx;
    if (!idx_include_halo) {
      real_idx = real_idx + m_tensor->get_halo_width();
    }
    return get_offset(
        real_idx, get_local_real_shape(),
        m_tensor->get_pitch());
  }

  LocaleMPI get_sub_locale(int dim) const {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    MPI_Comm sub_comm;
    const auto &dist = m_tensor->get_distribution();
    auto proc_idx = m_proc_idx;
    proc_idx[dim] = 0;
    int sub_comm_key = get_offset(proc_idx, dist.get_locale_shape());
    util::MPIPrintStreamDebug() << "sub_locale comm key: " << sub_comm_key
                                << ", proc_idx: " << proc_idx
                                << ", rank: " << m_tensor->get_locale().get_rank()
                                << ", locale shape: " << dist.get_locale_shape()
                                << ", tensor shape: " << m_tensor->get_shape();
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    return LocaleMPI(sub_comm, true);
  }

  LocaleMPI get_sub_locale_except_dim(int dim) const {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    MPI_Comm sub_comm;
    const auto &dist = m_tensor->get_distribution();
    auto proc_idx = m_proc_idx;
    dim = dim < 0 ? proc_idx.length() + dim : dim;
    for (int i = 0; i < proc_idx.length(); ++i) {
      if (i != dim) { proc_idx[i] = 0; }
    }
    int sub_comm_key = get_offset(proc_idx, dist.get_locale_shape());
    util::MPIPrintStreamDebug() << "sub_locale_except_dim comm key: " << sub_comm_key
                                << ", m_proc_idx: " << m_proc_idx
                                << ", proc_idx: " << proc_idx
                                << ", rank: " << m_tensor->get_locale().get_rank()
                                << ", locale shape: " << dist.get_locale_shape()
                                << ", tensor shape: " << m_tensor->get_shape();
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    return LocaleMPI(sub_comm, true);
  }

  LocaleMPI get_spatial_locale() const {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    MPI_Comm sub_comm;
    const auto &dist = m_tensor->get_distribution();
    auto proc_idx = m_proc_idx;
    for (int i = 0; i < get_num_spatial_dims(); ++i) {
      proc_idx[i] = 0;
    }
    int sub_comm_key = get_offset(proc_idx, dist.get_locale_shape());
    util::MPIPrintStreamDebug() << "sub comm key: " << sub_comm_key;
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    return LocaleMPI(sub_comm, true);
  }

  LocaleMPI get_split_sub_locale() const {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    MPI_Comm sub_comm;
    const auto &dist = m_tensor->get_distribution();
    auto split_idx = m_split_idx;
    int sub_comm_key = get_offset(split_idx, dist.get_split_shape());
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    return LocaleMPI(sub_comm, true);
  }

  LocaleMPI get_split_sub_locale(int dim) const {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    MPI_Comm sub_comm;
    const auto &dist = m_tensor->get_distribution();
    auto split_idx = m_split_idx;
    split_idx[dim] = 0;
    int sub_comm_key = get_offset(split_idx, dist.get_split_shape());
    util::MPIPrintStreamDebug() << "subcomm key: " << sub_comm_key << std::endl;
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    return LocaleMPI(sub_comm, true);
  }

  index_t get_dimension_rank_offset(int dim, int rank) const {
    return m_offset_all[dim][rank];
  }

  void allreduce_shared_regions() {
    auto subloc = m_tensor->get_split_sub_locale();
    DISTCONV_CHECK_MPI(
        MPI_Allreduce(MPI_IN_PLACE, m_tensor->get_buffer(),
                      m_tensor->get_local_pitched_size(),
                      util::get_mpi_data_type<DataType>(),
                      MPI_SUM, subloc.get_comm()));
  }

  void allreduce(const std::vector<int> &dims) {
    MPI_Comm comm = m_tensor->get_locale().get_comm();
    const auto &dist = m_tensor->get_distribution();
    auto sub_comm_idx = get_proc_index();
    for (auto d: dims) {
      sub_comm_idx[d] = 0;
    }
    int sub_comm_key = get_offset(sub_comm_idx, dist.get_locale_shape());
    MPI_Comm sub_comm;
    DISTCONV_CHECK_MPI(MPI_Comm_split(comm, sub_comm_key,
                                      m_tensor->get_locale().get_rank(),
                                      &sub_comm));
    DISTCONV_CHECK_MPI(
        MPI_Allreduce(MPI_IN_PLACE, m_tensor->get_buffer(),
                      m_tensor->get_local_pitched_size(),
                      util::get_mpi_data_type<DataType>(),
                      MPI_SUM, sub_comm));
    MPI_Comm_free(&sub_comm);
  }

  void scale(DataType v, typename Stream<Allocator>::type stream) {
    HelperType(*this).scale(v, stream);
  }

  void clear_halo(int dim, typename Stream<Allocator>::type stream) {
    HelperType(*this).clear_halo(dim, stream);
  }

  TensorType *get_tensor() {
    return m_tensor;
  }

  IndexVector get_proc_index() const {
    return m_proc_idx;
  }

  IndexVector get_split_index() const {
    return m_split_idx;
  }

 protected:

  int get_num_dims() const {
    return m_tensor->get_num_dims();
  }

  int get_num_spatial_dims() const {
    return m_tensor->get_num_spatial_dims();
  }

  void ensure_no_cyclic_distribution(const Distribution &dist) const {
    for (int i = 0; i < get_num_dims(); ++i) {
      assert0(dist.get_block_size(i));
    }
  }

  void init_proc_grid() {
    const auto &dist = m_tensor->get_distribution();
    m_proc_idx = m_tensor->get_locale().get_rank_idx(dist);
    m_split_idx = m_tensor->get_locale().get_split_idx(dist);
  }

  void init_local_tensor() {
    if (m_tensor->get_size() == 0) {
      util::MPIPrintStreamDebug()
          << "Skip initialization of empty local tensor.";
      return;
    }

    const auto &tensor_shape = m_tensor->get_shape();
    auto &dist = m_tensor->get_distribution();
    const auto &split_shape = dist.get_split_shape();

    for (int i = 0; i < get_num_dims(); ++i) {
      size_t proc_chunk_size;
      size_t real_size_extra = 0;
      // Set the local shape with the requested shape.
      // TODO: this won't work if the requested local shape is indeed
      // 0, which can happen.
      if (m_tensor->m_requested_local_shape[i]) {
        proc_chunk_size = m_tensor->m_requested_local_shape[i];
        real_size_extra = dist.get_overlap(i) * 2;
        util::MPIPrintStreamDebug()
            << "shape requested: " << proc_chunk_size;
      } else if (dist.is_distributed(i)) {
        // Make sure each sub tensor has a size that is divisible by
        // bsize. The remainder is taken care by the last process.
        index_t bsize = m_tensor->m_requested_local_block[i];
        if (bsize == 0) bsize = 1;
        auto d = tensor_shape[i] / bsize;
        util::MPIPrintStreamDebug()
            << "tensor_shape[" << i << "]: " << tensor_shape[i]
            << ", bsize: " << bsize;
        assert0(tensor_shape[i] % bsize);
        proc_chunk_size = d / split_shape[i];
        auto rem = d % split_shape[i];
        util::MPIPrintStreamDebug()
            << "proc_chunk_size: " << proc_chunk_size
            << ", rem: " << rem;
        if (rem) {
          util::MPIPrintStreamDebug()
              << "Tensor shape not divisible at dimension " << i << ". "
              << "Tensor: " << tensor_shape
              << ", split shape: " << split_shape
              << ", block size: " << bsize;
          if (m_split_idx[i] < rem) {
            ++proc_chunk_size;
          }
        }
        proc_chunk_size *= bsize;
        // if this is the last process of this dimension, take care of
        // the remainder of block size
        if (m_split_idx[i] == split_shape[i] - 1) {
          proc_chunk_size += tensor_shape[i] % bsize;
        }
        // Add halo regions
        real_size_extra = dist.get_overlap(i) * 2;
      } else {
        util::MPIPrintStreamDebug()
            << "no partitioning on dimension " << i;
        proc_chunk_size = tensor_shape[i];
        dist.set_overlap(i, 0);
      }
      m_local_shape[i] = proc_chunk_size;
      m_local_real_shape[i] = proc_chunk_size + real_size_extra;
    }
    util::MPIPrintStreamDebug()
        << "Tensor shape set. Global shape: "
        << m_tensor->get_shape()
        << ", local shape: " << m_local_shape
        << ", local real shape: " << m_local_real_shape;
    // offset requires scan of each local shape
    //init_offsets();
  }

  void init_offsets() {
    const auto &dist = m_tensor->get_distribution();
    const auto &loc_shape = dist.get_locale_shape();

    m_offset_all.clear();
    m_offset = IndexVector(get_num_dims(), 0);
    // Initialize max shape with the local shpae. Update as necessary
    m_max_local_shape = m_local_shape;
    for (int i = 0; i < get_num_dims(); ++i) {
      std::vector<index_t> offsets(loc_shape[i]);
      if (!dist.is_distributed(i)) {
        for (auto &offset: offsets) {
          offset = 0;
        }
      } else {
        LocaleMPI subloc = get_sub_locale(i);
        index_t local_dim = m_local_shape[i];
        MPI_Allgather(&local_dim, sizeof(index_t), MPI_BYTE,
                      offsets.data(), sizeof(index_t), MPI_BYTE,
                      subloc.get_comm());
        //util::MPIPrintStreamDebug() << "loc shape: " << loc_shape << "\n";
        // Scan
        index_t num_ranks_per_split = dist.get_num_ranks_per_split(i);
        index_t cur_offset = 0;
        index_t next_offset = 0;
        for (index_t j = 0; j < loc_shape[i]; j += num_ranks_per_split) {
          next_offset = cur_offset + offsets[j];
          for (index_t k = 0; k < num_ranks_per_split; ++k) {
            offsets[j+k] = cur_offset;
          }
          m_max_local_shape[i] = std::max(m_max_local_shape[i], next_offset - cur_offset);
          cur_offset = next_offset;
        }
        // The total size of dimension must match the size of the tensor
        if (next_offset != m_tensor->get_shape()[i]) {
          util::MPIPrintStreamError()
              << "The total size of dimension does not match the size of the tensor. "
              << "Dim: " << i
              << ", computed: " << next_offset
              << ", tensor shape: " << m_tensor->get_shape()[i]
              << ", local shape: " << m_tensor->get_local_shape()[i]
              << ", locale shape: " << loc_shape
              << ", split shape: " << dist.get_split_shape();
          std::abort();
        }
      }
      m_offset_all.push_back(offsets);
      m_offset[i] = offsets[m_proc_idx[i]];
#if 0
      std::stringstream ss;
      util::print_vector(ss, m_offset_all.back().begin(), m_offset_all.back().end());
      util::MPIPrintStreamDebug()
          << "Tensor offsets for dimension " << i << ": " << ss.str();
#endif
    }
  }

  TensorType *m_tensor = nullptr;
  IndexVector m_proc_idx;
  IndexVector m_split_idx;
  Shape m_local_shape;
  Shape m_local_real_shape;
  Shape m_max_local_shape;
  IndexVector m_offset;
  std::vector<std::vector<index_t>> m_offset_all;
};

namespace internal {

// Use a partially specialized functor so that its generic version is
// specified as a friend of class Tensor. Furthermore, to make it
// available as a function, declare a function interface too.
template <typename DataType, typename Allocator>
struct ViewFunctor<Tensor<DataType, LocaleProcess, Allocator>,
                   Tensor<DataType, LocaleMPI, Allocator>> {
  int operator()(Tensor<DataType, LocaleProcess, Allocator> &t_proc,
                 const Tensor<DataType, LocaleMPI, Allocator> &t_mpi) {
    assert_always(t_proc.is_null());
    assert_always(t_proc.get_shape().is_empty());

    t_proc.set_shape(t_mpi.get_local_shape());
    assert_eq(t_mpi.get_local_shape(), t_proc.get_shape());

    // MPI local region may have halo
    auto dist = t_proc.get_distribution();
    dist.copy_overlap(t_mpi.get_distribution());
    t_proc.set_distribution(dist);
    assert_eq(t_mpi.get_local_real_shape(), t_proc.get_local_real_shape());
    t_proc.set_view(t_mpi.m_data);
    return 0;
  }
};

template <typename DataType, typename Allocator>
struct ViewFunctor<Tensor<DataType, LocaleMPI, Allocator>,
                   Tensor<DataType, LocaleMPI, Allocator>> {
  int operator()(Tensor<DataType, LocaleMPI, Allocator> &t_viewer,
                 const Tensor<DataType, LocaleMPI, Allocator> &t_original) {
    t_viewer.set_view(t_original.get_data());
    t_viewer.m_locale = t_original.get_locale();
    t_viewer.set_distribution(t_original.get_distribution());
    t_viewer.m_requested_local_block = t_original.get_requested_local_block();
    t_viewer.m_requested_local_shape = t_original.get_requested_local_shape();
    t_viewer.set_shape(t_original.get_shape());
    util::MPIPrintStreamDebug()
        << "View created. original: " << t_original
        << ", viewer: " << t_viewer;
    return 0;
  }
};

} // namespace internal

template <typename DataType, typename Allocator>
inline int View(Tensor<DataType, LocaleProcess, Allocator> &t_proc,
                const Tensor<DataType, LocaleMPI, Allocator> &t_mpi) {
  return internal::ViewFunctor<Tensor<DataType, LocaleProcess, Allocator>,
                               Tensor<DataType, LocaleMPI, Allocator>>()(t_proc, t_mpi);
}

template <typename DataType, typename Allocator>
inline int View(Tensor<DataType, LocaleMPI, Allocator> &t, DataType *raw_ptr) {
  t.set_view(raw_ptr);
  return 0;
}

// Note: raw_ptr should not be pitched memory
template <typename DataType, typename Allocator>
inline int View(Tensor<DataType, LocaleMPI, Allocator> &t,
                const DataType *raw_ptr) {
  t.set_view(raw_ptr);
  return 0;
}

template <typename DataType, typename Allocator>
inline int View(
    Tensor<DataType, LocaleMPI, Allocator> &t_viewer,
    const Tensor<DataType, LocaleMPI, Allocator> &t_original) {
  return internal::ViewFunctor<Tensor<DataType, LocaleMPI, Allocator>,
                               Tensor<DataType, LocaleMPI, Allocator>>()(
                                   t_viewer, t_original);
}

namespace internal {

template <typename DataType, typename AllocatorProc, typename AllocatorMPI>
struct CopyFunctor<Tensor<DataType, LocaleProcess, AllocatorProc>,
                   Tensor<DataType, LocaleMPI, AllocatorMPI>> {
  using TensorProcType = Tensor<DataType, LocaleProcess, AllocatorProc>;
  using TensorMPIType = Tensor<DataType, LocaleMPI, AllocatorMPI>;

  DataType *m_buf = nullptr;
  size_t m_size = 0;

  void ensure_buffer(size_t size) {
    if (m_size < size) {
      if (m_buf) {
        free(m_buf);
      }
      m_buf = (DataType*)malloc(size);
      m_size = size;
    }
  }

  ~CopyFunctor() {
    if (m_buf) {
      free(m_buf);
    }
  }

  void copy_into_local_buffer(DataType *dest, const DataType *src,
                              const Shape &local_shape,
                              const IndexVector &global_offset,
                              const Shape &global_shape,
                              const IndexVector &overlap,
                              size_t pitch) {
    auto local_real_shape = local_shape + overlap * 2;
    // copy the MPI local buffer to the destination tensor
    for (auto it = local_shape.index_begin();
         it != local_shape.index_end();) {
      auto src_offset = get_offset(
          *it + overlap, local_real_shape, pitch);
      auto dest_offset = get_offset(global_offset + *it,
                                       global_shape);
      memcpy(dest + dest_offset,
             src + src_offset, local_shape[0] * sizeof(DataType));
      // Skip the rest of the points at dimension 0
      for (index_t i = 0; i < local_shape[0]; ++i) {
        ++it;
      }
    }
  }

  template <typename TensorMPIType>
  void send_local_buffer(const TensorMPIType &t_mpi, int root,
                         int tag) {
    const int nd = t_mpi.get_num_dims();
    // Send offset and shape
    IndexVector offset = t_mpi.get_global_index();
    //util::PrintStreamDebug() << "Sending offset to " << root;
    MPI_Send(offset.data(), sizeof(IndexVector::data_type) * nd, MPI_BYTE,
             root, tag, t_mpi.m_locale.get_comm());
    Shape shape = t_mpi.get_local_shape();
    MPI_Send(shape.data(), sizeof(Shape::data_type) * nd, MPI_BYTE, root,
             tag, t_mpi.m_locale.get_comm());
    if (shape.get_size() == 0) {
      util::MPIPrintStreamDebug() << "Empty tensor. Not sending";
      return;
    }
    // Send the buffer
    const DataType *send_buffer = t_mpi.get_const_buffer();
    size_t buffer_size = t_mpi.m_data.get_real_size();

    MPI_Send(&buffer_size, sizeof(size_t), MPI_BYTE, root,
             tag, t_mpi.m_locale.get_comm());
    assert_always(send_buffer);
    // MVAPICH2-2.3rc1 seems to be hanging up here.
#if 0
    MPI_Send(send_buffer, buffer_size, MPI_BYTE, root,
             tag, t_mpi.m_locale.get_comm());
#else
    DataType *host_buf = (DataType*)malloc(buffer_size);
    assert_always(host_buf);
    t_mpi.m_data.copyout(host_buf);
    MPI_Send(host_buf, buffer_size, MPI_BYTE, root,
             tag, t_mpi.m_locale.get_comm());
    free(host_buf);
#endif
    size_t pitch = t_mpi.get_pitch();
    MPI_Send(&pitch, sizeof(size_t), MPI_BYTE, root,
             tag, t_mpi.m_locale.get_comm());
  }

  template <typename TensorProcType, typename TensorMPIType>
  void recv_local_buffer(TensorProcType &t_proc,
                         const TensorMPIType &t_mpi, int src,
                         int tag) {
    const int nd = t_proc.get_num_dims();
    int my_rank = t_mpi.m_locale.get_rank();
    const DataType *src_buf = nullptr;
    IndexVector global_offset(nd);
    Shape shape(nd);
    size_t pitch;
    if (src == my_rank) {
      util::MPIPrintStreamDebug()
          << "Buffer size: " << t_mpi.m_data.get_real_size()
          << ", local real size: " << t_mpi.get_local_real_size();
      ensure_buffer(t_mpi.m_data.get_real_size());
      t_mpi.m_data.copyout(m_buf);
      global_offset = t_mpi.get_global_index();
      shape = t_mpi.get_local_shape();
      pitch = t_mpi.get_pitch();
    } else {
      MPI_Recv(global_offset.data(), sizeof(IndexVector::data_type) * nd,
               MPI_BYTE, src, tag, t_mpi.m_locale.get_comm(),
               MPI_STATUS_IGNORE);
      MPI_Recv(shape.data(), sizeof(Shape::data_type) * nd, MPI_BYTE, src,
               tag, t_mpi.m_locale.get_comm(), MPI_STATUS_IGNORE);
      if (shape.get_size() == 0) {
        util::MPIPrintStreamDebug() << "Empty tensor. Not receiving";
        return;
      }
      size_t buffer_size;
      MPI_Recv(&buffer_size, sizeof(size_t), MPI_BYTE, src,
               tag, t_mpi.m_locale.get_comm(), MPI_STATUS_IGNORE);
      // Receiving the source local tensor
      ensure_buffer(buffer_size);
      MPI_Recv(m_buf, buffer_size, MPI_BYTE, src,
               tag, t_mpi.m_locale.get_comm(), MPI_STATUS_IGNORE);
      MPI_Recv(&pitch, sizeof(size_t), MPI_BYTE, src,
               tag, t_mpi.m_locale.get_comm(), MPI_STATUS_IGNORE);
    }
    src_buf = m_buf;
    const auto &overlap = t_mpi.get_halo_width();
    copy_into_local_buffer(t_proc.get_buffer(), src_buf, shape,
                           global_offset, t_mpi.get_shape(), overlap, pitch);
  }

  int operator()(TensorProcType &t_proc, const TensorMPIType &t_mpi,
                 int root) {
    util::MPIPrintStreamDebug()
        << "Gathering " << t_mpi << " to "
        << t_proc << " at proc " << root;

    int num_ranks = t_mpi.m_locale.get_size();
    int my_rank = t_mpi.m_locale.get_rank();

    // Make sure the destination tensor is empty
    assert_always(t_proc.is_null());
    assert_always(t_proc.get_shape().is_empty());

    // Set the destination tensor as the same size of the global shape
    // of the source tensor
    if (my_rank == root) {
      t_proc.set_shape(t_mpi.get_shape());
      //util::PrintStreamDebug() << "Root shape: " << t_proc.get_shape();
      assert_eq(t_mpi.get_shape(), t_proc.get_shape());
      assert0(t_proc.allocate());
    }

    // Copy each local tensor to the root sequentially
    int tag = 0;
    if (my_rank == root) {
      for (int src = 0; src < num_ranks; ++src) {
        recv_local_buffer(t_proc, t_mpi, src, tag);
      }
    } else {
      send_local_buffer(t_mpi, root, tag);
    }
    return 0;
  }
};

template <typename DataType, typename Allocator>
void find_owning_process(const Tensor<DataType, LocaleMPI, Allocator> &tensor,
                         const IndexVector &global_idx,
                         IndexVector &rank,
                         IndexVector &local_offset) {
  const int nd = tensor.get_num_dims();
  auto dist = tensor.get_distribution();
  if (!dist.is_distributed()) {
    rank = tensor.get_locale().get_rank_idx(dist);
    local_offset = global_idx;
    return;
  }

  auto loc_shape = dist.get_locale_shape();
  rank = IndexVector(nd);
  local_offset = IndexVector(nd);
  for (int i = 0; i < nd; ++i) {
    int rank_idx = 0;
    for (index_t j = 1; j < loc_shape[i]; ++j) {
      if (global_idx[i] < tensor.get_dimension_rank_offset(i, j)) {
        rank_idx = j - 1;
        break;
      }
      rank_idx = j;
    }
    // we now found it's owned by process at rank_idx
    rank[i] = rank_idx;
    local_offset[i] = global_idx[i] - tensor.get_dimension_rank_offset(i, rank_idx);
  }
}

// REFACTORING: this should be cleaned up
template <typename Tensor>
struct HostShadow;

template <typename DataType>
struct HostShadow<Tensor<DataType, LocaleMPI, BaseAllocator>> {
  using TensorType = Tensor<DataType, LocaleMPI, BaseAllocator>;
  using ShadowTensorType = Tensor<DataType, LocaleMPI, BaseAllocator>;

  HostShadow() = delete;
  HostShadow(const TensorType &tensor): m_tensor(tensor) {}

  const ShadowTensorType &get_host_shadow() const {
    return m_tensor;
  }

  ShadowTensorType &get_host_shadow() {
    return m_tensor;
  }

  void sync_to_dev() {}
  void sync_from_dev() {}

 protected:
  TensorType m_tensor;
};

template <typename DataType, typename AllocSrc, typename AllocDest,
          typename StreamType>
int CopyByShuffle(Tensor<DataType, LocaleMPI, AllocDest> &t_dest,
                  const Tensor<DataType, LocaleMPI, AllocSrc> &t_src,
                  StreamType stream) {
  util::MPIPrintStreamWarning() << "CopyByShuffle: "
                                << t_dest << " <- " << t_src;

  using TensorSrcType = Tensor<DataType, LocaleMPI, AllocSrc>;
  using TensorDestType = Tensor<DataType, LocaleMPI, AllocDest>;

  HostShadow<TensorDestType> t_dest_shadow(t_dest);
  typename HostShadow<TensorDestType>::ShadowTensorType &t_dest_host =
      t_dest_shadow.get_host_shadow();
  HostShadow<TensorSrcType> t_src_shadow(t_src);
  t_src_shadow.sync_from_dev();
  typename HostShadow<TensorSrcType>::ShadowTensorType &t_src_host =
      t_src_shadow.get_host_shadow();

  index_t *pitch_sizes = new index_t[t_dest_host.get_locale().get_size()];
  index_t self_pitch = t_dest_host.get_pitch();
  DISTCONV_CHECK_MPI(MPI_Allgather(&self_pitch, sizeof(index_t), MPI_BYTE,
                                   pitch_sizes, sizeof(index_t), MPI_BYTE,
                                   t_dest_host.get_locale().get_comm()));

  // Open the buffer area of t_dest as an MPI RMA window
  MPI_Win win;
  typename TensorSrcType::data_type *src_buf = t_src_host.get_buffer();
  typename TensorDestType::data_type *dest_buf = t_dest_host.get_buffer();
#if 0
  util::MPIPrintStreamDebug()
      << "win_create. "
      << "buf: " << dest_buf
      << ", original dest: " << t_dest.get_buffer()
      << ", local shape: " << t_dest_host.get_local_real_shape()
      << ", size: " << t_dest_host.get_local_real_size() * sizeof(DataType)
      << ", logical size: " << t_dest_host.get_local_size() * sizeof(DataType)
      << ", original size: " << t_dest.get_local_real_size() * sizeof(DataType);
#endif
  MPI_Win_create(dest_buf, t_dest_host.get_local_real_size() * sizeof(DataType),
                 1, MPI_INFO_NULL, t_dest_host.get_locale().get_comm(),
                 &win);
  MPI_Win_fence(0, win);

  const auto src_shape = t_src_host.get_local_shape();
  auto loc_shape = t_dest_host.get_distribution().get_locale_shape();
  int put_count = 0;
  // delay puts for continuous elements
  index_t cur_src_offset = 0;
  // set -1 to indicate nothing delayed yet
  int cur_target_rank = -1;
  index_t cur_target_offset = 0;
  int cur_bytes = 0;
  for (auto it = src_shape.index_begin();
       it != src_shape.index_end(); ++it) {
    index_t src_offset = t_src_host.get_local_offset(*it);
    auto global_idx = t_src_host.get_global_index(*it);
    IndexVector target_rank_idx;
    IndexVector target_local_idx;
    find_owning_process(t_dest_host, global_idx, target_rank_idx, target_local_idx);
    int target_rank = get_offset(target_rank_idx, loc_shape);
    auto target_local_idx_with_halo = target_local_idx +
        t_dest_host.get_halo_width();
    auto remote_shape = t_dest_host.get_remote_real_shape(target_rank_idx);
    index_t target_offset = get_offset(target_local_idx_with_halo, remote_shape,
                                       pitch_sizes[target_rank]) * sizeof(DataType);

    // check if it's still done with a continuous send
    if (src_offset == cur_src_offset + cur_bytes / sizeof(DataType) &&
        target_rank == cur_target_rank &&
        target_offset == cur_target_offset + cur_bytes) {
      cur_bytes += sizeof(DataType);
    } else {
      if (cur_bytes != 0) {
#if 0
        util::MPIPrintStreamDebug()
            << "Putting " << cur_bytes << " bytes from "
            << src_offset << " at local idx: " << *it
            << ", global idx: " << global_idx
            << " to rank "
            << target_rank << " at offset " << target_local_idx
            << " (" << target_offset << "), "
            << target_local_idx_with_halo << ", "
            << "remote shape: " << remote_shape
            << ", pitch: " << pitch_sizes[target_rank];
#endif
        // Use std::memcpy for local copies
        if (cur_target_rank == t_src_host.get_locale().get_rank()) {
          std::memcpy(dest_buf + cur_target_offset / sizeof(DataType),
                      src_buf + cur_src_offset,
                      cur_bytes);
        } else {
          MPI_Put(src_buf + cur_src_offset, cur_bytes, MPI_BYTE,
                  cur_target_rank, cur_target_offset,
                  cur_bytes, MPI_BYTE, win);
          ++put_count;
        }
      }
      cur_src_offset = src_offset;
      cur_target_rank = target_rank;
      cur_target_offset = target_offset;
      cur_bytes = sizeof(DataType);
    }
  }

  // needs to flush the remaining elements
  if (cur_bytes != 0) {
    MPI_Put(src_buf + cur_src_offset, cur_bytes, MPI_BYTE,
            cur_target_rank, cur_target_offset,
            cur_bytes, MPI_BYTE, win);
  }

  MPI_Win_fence(0, win);
  MPI_Win_free(&win);

  util::MPIPrintStreamDebug() << "#put calls: " << put_count;

  t_dest_shadow.sync_to_dev();
  return 0;
}

template <typename DataType, typename AllocSrc,
          typename AllocDest, typename StreamType>
struct CopyLocalFunctor {
  int operator()(Tensor<DataType, LocaleMPI, AllocDest> &t_dst,
                 const Tensor<DataType, LocaleMPI, AllocSrc> &t_src,
                 StreamType stream) {
    const int nd = t_src.get_num_dims();
    const auto local_shape = t_src.get_local_shape();
    assert_eq(local_shape, t_dst.get_local_shape());
    assert_eq(nd, t_dst.get_num_dims());
    assert_always(nd >= 2);
    auto tr_shape = local_shape;
    if (t_src.get_halo_width(0) == 0 && t_src.get_halo_width(1) == 0 &&
        t_dst.get_halo_width(0) == 0 && t_dst.get_halo_width(1) == 0 &&
        nd >= 3) {
      return copy_opt(t_dst, t_src, stream);
    }
    // Use the 2D copy feature of Memory for the first 2 dimensions
    tr_shape[0] = 1;
    tr_shape[1] = 1;
    for (auto it = tr_shape.index_begin(); it != tr_shape.index_end();
         ++it) {
      Copy(t_dst.get_data(), t_src.get_data(),
           local_shape[0] * sizeof(DataType),
           local_shape[1],
           (t_dst.get_local_offset(*it) + t_dst.get_halo_width(0))
           * sizeof(DataType),
           t_dst.get_halo_width(1),
           (t_src.get_local_offset(*it) + t_src.get_halo_width(0))
           * sizeof(DataType),
           t_src.get_halo_width(1),
           stream);
    }
    return 0;
  }

  int copy_opt(Tensor<DataType, LocaleMPI, AllocDest> &t_dst,
               const Tensor<DataType, LocaleMPI, AllocSrc> &t_src,
               StreamType stream) {
    const auto local_shape = t_src.get_local_shape();
    auto tr_shape = local_shape;
    tr_shape[0] = 1;
    tr_shape[1] = 1;
    tr_shape[2] = 1;
    for (auto it = tr_shape.index_begin(); it != tr_shape.index_end();
         ++it) {
      auto x_len = local_shape[0] * sizeof(DataType);
      auto y_len = local_shape[1] * local_shape[2];
      Copy(t_dst.get_data(), t_src.get_data(),
           x_len, y_len, t_dst.get_local_offset(*it) * sizeof(DataType), 0,
           t_src.get_local_offset(*it) * sizeof(DataType), 0,
           stream);
    }
    return 0;
  }
};

// Use just CopyLocalFunctor by default.
template <typename DataType, typename AllocSrc, typename AllocDest,
          typename StreamType>
struct CopyLocalFunctor3D {
  int operator()(Tensor<DataType, LocaleMPI, AllocDest> &t_dst,
                 const Tensor<DataType, LocaleMPI, AllocSrc> &t_src,
                 StreamType stream) {
    return CopyLocalFunctor<DataType, AllocSrc, AllocDest,
                            StreamType>()(t_dst, t_src, stream);
  }
};

} // namespace internal

template <typename DataType, typename AllocatorProc,
          typename AllocatorMPI, typename StreamType=DefaultStream>
inline int Copy(Tensor<DataType, LocaleProcess, AllocatorProc> &t_proc,
                const Tensor<DataType, LocaleMPI, AllocatorMPI> &t_mpi,
                int root, StreamType stream=DefaultStream::value) {
  return internal::CopyFunctor<
    Tensor<DataType, LocaleProcess, AllocatorProc>,
    Tensor<DataType, LocaleMPI, AllocatorMPI>>()(
        t_proc, t_mpi, root);
}

template <typename DataType, typename AllocSrc, typename AllocDest,
          typename StreamType=DefaultStream>
inline int Copy(Tensor<DataType, LocaleMPI, AllocDest> &t_dest,
                const Tensor<DataType, LocaleMPI, AllocSrc> &t_src,
                StreamType stream=DefaultStream::value) {
  const int nd = t_src.get_num_dims();
  util::MPIPrintStreamDebug() <<
      "Copying " << t_src << " to " << t_dest;

  // tensor shape must match
  if (t_dest.get_shape() != t_src.get_shape()) {
    util::MPIPrintStreamError()
        << "Can't copy between tensors with different shapes";
    return 1;
  }

  if (t_dest.is_null() && !t_dest.get_local_shape().is_empty()) {
    util::MPIPrintStreamDebug()
        << "Dest tensor is null. Allocating tensor";
    assert0(t_dest.allocate());
    assert_always(!t_dest.is_null());
  }

  // if both tensors use the same distribution, just copy the local
  // tensor at each process
  if (t_dest.get_distribution() == t_src.get_distribution()) {
    if (t_dest.get_local_shape().is_empty() ||
        t_src.get_local_shape().is_empty()) {
      return 0;
    }
    auto local_shape = t_src.get_local_real_shape();
    index_t y_len = local_shape.get_size() / local_shape[0];
    util::MPIPrintStreamDebug()
        << "Tensors are exact same shape. Using normal linear copy";
    return Copy(t_dest.get_data(), t_src.get_data(),
                local_shape[0] * sizeof(DataType), y_len,
                0, 0, 0, 0, stream);
  }

  // if locale shape is the same, but the halo size is different, copy
  // can be done locally
  if ((t_dest.get_distribution().get_locale_shape() ==
       t_src.get_distribution().get_locale_shape()) &&
      (t_dest.get_distribution().get_split_shape() ==
       t_src.get_distribution().get_split_shape())) {
    if (t_dest.get_local_shape().is_empty() ||
        t_src.get_local_shape().is_empty()) {
      return 0;
    }
    util::MPIPrintStreamDebug()
        << "Tensor distributions are the same, but different halo widths.";
    //internal::CopyLocalFunc(t_dest, t_src);
    if (nd >= 3) {
      internal::CopyLocalFunctor3D<DataType, AllocSrc, AllocDest,
                                   StreamType>()(t_dest, t_src, stream);
    } else {
      internal::CopyLocalFunctor<DataType, AllocSrc, AllocDest,
                                 StreamType>()(t_dest, t_src, stream);
    }
    return 0;
  }

  util::MPIPrintStreamDebug() << "Falling back to CopyByShuffle";

  // the tensors are distributed differently. use copy_by_shuffle
  return internal::CopyByShuffle(t_dest, t_src, stream);
}

} // namespace tensor
} // namespace distconv
