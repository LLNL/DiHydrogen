#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <array>
#include <iterator>
#include <sstream>
#include <iostream>
#include <exception>

#include "distconv/tensor/tensor_base.hpp"
#include "distconv/tensor/memory.hpp"
#include "distconv/tensor/distribution.hpp"
#include "distconv/util/util.hpp"

namespace distconv {
namespace tensor {

template <typename TensorType>
class TensorImpl;

namespace internal {
template <typename TensorTypeX, typename TensorTypeY>
struct ViewFunctor;

template <typename TensorTypeX, typename TensorTypeY>
struct CopyFunctor;

} // namespace internal

template <typename DataType, typename Locale,
          typename Allocator>
class Tensor {
  using TensorImplType = TensorImpl<Tensor<DataType, Locale, Allocator>>;
  friend TensorImplType;
  template <typename TensorTypeX, typename TensorTypeY>
  friend struct internal::ViewFunctor;
  template <typename TensorTypeX, typename TensorTypeY>
  friend struct internal::CopyFunctor;
  int m_num_dims;
 public:

  using data_type = DataType;
  using locale_type = Locale;
  using allocator_type = Allocator;

  Tensor(): Tensor(Locale(), Distribution()) {}

  Tensor(const Locale &locale, const Distribution &dist):
      Tensor(Shape(dist.num_dims(), 0), locale, dist) {}

  Tensor(const Shape &shape, const Locale &locale,
         const Distribution &dist):
      Tensor(shape, locale, dist, Shape(shape.num_dims(), 0)) {}

  Tensor(const Shape &shape, const Locale &locale,
         const Distribution &dist,
         const Shape &requested_local_shape):
      Tensor(shape, locale, dist, requested_local_shape,
             Shape(shape.num_dims(), 1)) {}

  Tensor(const Shape &shape, const Locale &locale,
         const Distribution &dist,
         const Shape &requested_local_shape,
         const Shape &requested_local_block):
      m_num_dims(shape.num_dims()),
      m_shape(shape), m_requested_local_shape(requested_local_shape),
      m_requested_local_block(requested_local_block),
      m_locale(locale), m_dist(dist), m_data(), m_impl(this)  {
    check_shape_validity();
  }

  virtual ~Tensor() = default;

  Tensor &operator=(const Tensor &t) {
    m_num_dims = t.get_num_dims();
    m_shape = t.m_shape;
    m_requested_local_shape = t.m_requested_local_shape;
    m_requested_local_block = t.m_requested_local_block;
    m_locale = t.m_locale;
    m_dist = t.m_dist;
    m_is_view = t.m_is_view;
    m_data = t.m_data;
    m_impl = TensorImplType(this, t.m_impl);
    check_shape_validity();
    return *this;
  }

  Tensor(const Tensor &t):
      m_num_dims(t.get_num_dims()),
      m_shape(t.m_shape), m_requested_local_shape(t.m_requested_local_shape),
      m_requested_local_block(t.m_requested_local_block),
      m_locale(t.m_locale), m_dist(t.m_dist),
      m_is_view(t.m_is_view), m_data(t.m_data),
      m_impl(this, t.m_impl) {
    check_shape_validity();
  }

  void set_outermost_dimension(index_t d) {
    m_impl.set_outermost_dimension(d);
  }

  // This should not be used as much as possible
  TensorImplType &get_impl() {
    return m_impl;
  }

  // This should not be used as much as possible
  const TensorImplType &get_impl() const {
    return m_impl;
  }

  constexpr int get_num_dims() const {
    return m_num_dims;
  }

  constexpr int get_num_spatial_dims() const {
    return get_num_dims() - 2;
  }

  const Locale &get_locale() const {
    return m_locale;
  }

  Locale get_sub_locale(int dim) const {
    return m_impl.get_sub_locale(dim);
  }

  // Like get_sub_locale, except the communicator contains all ranks *except*
  // those with the same index on dim.
  // TODO: Better name?
  Locale get_sub_locale_except_dim(int dim) const {
    return m_impl.get_sub_locale_except_dim(dim);
  }

  Locale get_spatial_locale() const {
    return m_impl.get_spatial_locale();
  }

  // Returns a communicator with the same split index
  Locale get_split_sub_locale() const {
    return m_impl.get_split_sub_locale();
  }

  // Returns a communicator with locales along dimension dim and with
  // the same split index
  Locale get_split_sub_locale(int dim) const {
    return m_impl.get_split_sub_locale(dim);
  }

  const Memory<Allocator> &get_data() const {
    return m_data;
  }

  Memory<Allocator> &get_data() {
    return m_data;
  }

  const Shape &get_shape() const {
    return m_shape;
  }

  Shape get_local_shape() const {
    return m_impl.get_local_shape();
  }

  Shape get_local_real_shape() const {
    return m_impl.get_local_real_shape();
  }

  index_t get_local_real_size() const {
    return get_local_real_shape().size();
  }

  // includes the overlap region
  Shape get_local_pitched_shape() const {
    auto s = get_local_real_shape();
    s[0] = get_pitch();
    return s;
  }

  // includes the overlap region
  index_t get_local_pitched_size() const {
    return get_local_pitched_shape().size();
  }

  Shape get_max_local_shape() const {
    return m_impl.get_max_local_shape();
  }

  Shape get_max_local_real_shape() const {
    return m_impl.get_max_local_real_shape();
  }

  /**
     Converts a local index to global.
     @param dim dimension.
     @param local_idx local index.
     @return the global index of the local index.
   */
  index_t get_global_index(int dim, index_t local_idx) const {
    return m_impl.get_global_index(dim, local_idx);
  }

  /**
     Converts an array of local indices to global.
     @param local_idx local indices.
     @return the global index array of the local indices.
   */
  IndexVector get_global_index(const IndexVector &local_idx) const {
    IndexVector gi(get_num_dims());
    for (int i = 0; i < get_num_dims(); ++i) {
      gi[i] = m_impl.get_global_index(i, local_idx[i]);
    }
    return gi;
  }

  /**
     Return the global index array of the local base offset.
   */
  IndexVector get_global_index() const {
    return get_global_index(IndexVector(get_num_dims(), 0));
  }

  /**
     Return the offset within a tensor.
     @param local_idx an local index array.
     @return The linear global offset to the local index.
   */
  index_t get_global_offset(const IndexVector &local_idx) const {
    auto gi = get_global_index(local_idx);
    return get_offset(gi, get_shape());
  }

  /**
     Return the global offset to the local base index.
   */
  index_t get_global_offset() const {
    return get_global_offset(IndexVector(get_num_dims(), 0));
  }

  /**
     Converts an array of global indices to local.

     This does not check the global indices are located inside the
     local partition.

     @param global_idx a global index array.
     @return the local index array of the global indices.
   */
  IndexVector get_local_index(const IndexVector &global_idx) const {
    IndexVector li;
    for (int i = 0; i < get_num_dims(); ++i) {
      li.push_back(m_impl.get_local_index(i, global_idx[i]));
    }
    return li;
  }

  /**
     Get a linear offset within the local partition.

     @param idx a local index array.
     @param idx_include_halo indicates whether the base of the the.
     local index includes the halo space.
     @returns The linear offset to the given local index.
   */
  index_t get_local_offset(const IndexVector &local_idx,
                           bool idx_include_halo=false) const {
    assert_always(get_local_size() > 0);
    return m_impl.get_local_offset(local_idx, idx_include_halo);
  }

  /**
     Get a linear offset to the logical base of the local partition.

     This is always zero if the tensor does not have halo. Otherwise,
     it returns the offset to the logical base.
   */
  index_t get_local_offset() const {
    return get_local_offset(IndexVector(get_num_dims(), 0), false);
  }

  Shape get_strides() const {
    auto &&real_shape = get_local_pitched_shape();
    Shape strides(get_num_dims(), 1);
    for (int i = 1; i < get_num_dims(); ++i) {
      strides[i] = strides[i-1] * real_shape[i-1];
    }
    return strides;
  }

  // Returns the offset within dimension dim of sub tensor held by rank
  index_t get_dimension_rank_offset(int dim, int rank) const {
    return m_impl.get_dimension_rank_offset(dim, rank);
  }

  IndexVector get_remote_index(const IndexVector &rank_idx) const {
    IndexVector rs;
    for (int i = 0; i < get_num_dims(); ++i) {
      rs.push_back(get_dimension_rank_offset(i, rank_idx[i]));
    }
    return rs;
  }

  /**
     Return the dimension of a local tensor.

     The local tensor is not necessarily located locally.

     @param dim a dimension index
     @param rank_idx a rank index
     @return The dimension of the local tensor.
   */
  index_t get_remote_dimension(int dim, index_t rank_idx) const {
    auto loc_shape = get_locale_shape();
    index_t offset = get_dimension_rank_offset(dim, rank_idx);
    index_t next_offset = get_shape()[dim];
    // Locates the next rank that is not within the same split as
    // this rank.
    for (auto next = rank_idx+1; next < loc_shape[dim]; ++next) {
      if (get_distribution().is_split_root(dim, next)) {
        next_offset = get_dimension_rank_offset(dim, next);
        break;
      }
    }
    return next_offset - offset;
  }

  /**
     Get the local shape of a remote tensor.
     @param rank_idx an index array of a remote rank.
     @return the local shape.
   */
  Shape get_remote_shape(const IndexVector &rank_idx) const {
    Shape rs;
    for (int i = 0; i < get_num_dims(); ++i) {
      rs.push_back(get_remote_dimension(i, rank_idx[i]));
    }
    return rs;
  }

  /**
     Get the real shape of a remote tensor.

     Adds the extra dimension for halo data. Note that the halo size
     is assumed to be equal across ranks.

     @param rank_idx an index array of a remote rank.
     @return the local real shape.
   */
  Shape get_remote_real_shape(const IndexVector &rank_idx) const {
    auto rs = get_remote_shape(rank_idx);
    for (int i = 0; i < get_num_dims(); ++i) {
      rs[i] += get_overlap()[i] * 2;
    }
    return rs;
  }

  /**
     Get the pitched shape of a remote tensor.

     @param rank_idx an index array of a remote rank.
     @return the local pitched shape.
   */
  Shape get_remote_pitched_shape(const IndexVector &rank_idx) const {
    auto rs = get_remote_real_shape(rank_idx);
    rs[0] = get_pitch();
    return rs;
  }

  const Distribution &get_distribution() const {
    return m_dist;
  }

  Distribution &get_distribution() {
    return m_dist;
  }

  const IndexVector &get_locale_shape() const {
    return get_distribution().get_locale_shape();
  }

  const IndexVector &get_split_shape() const {
    return get_distribution().get_split_shape();
  }

  bool is_view() const {
    return m_is_view;
  }

  size_t get_size() const {
    return m_shape.size();
  }

  index_t get_local_size() const {
    return m_impl.get_local_size();
  }

  int allocate() {
    if (m_shape.is_empty()) {
      util::PrintStreamError() << "Empty shape";
      return -1;
    }
    if (m_dist.get_locale_shape().is_empty()) {
      util::PrintStreamError() <<
          "Empty locale shape: " << m_dist.get_locale_shape();
      return -1;
    }
    return m_impl.allocate();
  }

  int nullify() {
    m_impl.nullify();
    return 0;
  }

  bool is_null() const {
    return m_data.is_null();
  }

  bool is_non_null() const {
    return !m_data.is_null();
  }

  /*
   * The buffer must have the same size and other properties as they
   * were allocated by this class.
   */
  int attach(DataType *buffer) {
    return m_data.attach(buffer);
  }

  DataType get(const IndexVector &local_idx,
                bool idx_include_halo=false) const {
    index_t offset = get_local_offset(local_idx, idx_include_halo);
    const DataType *buf = get_const_buffer();
    assert_always(buf != nullptr);
    return buf[offset];
  }

  const DataType *get_const_buffer(bool logical_offset=false) const {
    if (get_local_size() == 0) return nullptr;
    const DataType *p = static_cast<const DataType*>(m_data.get());
    if (p != nullptr && logical_offset) {
      p += get_local_offset();
    }
    return p;
  }

  DataType *get_buffer(bool logical_offset=false) {
    if (get_local_size() == 0) return nullptr;
    DataType *p = static_cast<DataType*>(m_data.get());
    if (p != nullptr && logical_offset) {
      p += get_local_offset();
    }
    return p;
  }

  const DataType *get_buffer(bool logical_offset=false) const {
    return get_const_buffer(logical_offset);
  }

  const DataType *get_const_base_ptr() const {
    return get_const_buffer(true);
  }

  DataType *get_base_ptr() {
    return get_buffer(true);
  }

  const DataType *get_base_ptr() const {
    return get_const_buffer(true);
  }

  size_t get_pitch() const {
    return m_data.get_pitch() / sizeof(DataType);
  }

  void zero(typename Stream<Allocator>::type stream=
            Stream<Allocator>::default_value) {
    m_data.memset(0, stream);
  }

  void scale(DataType v, typename Stream<Allocator>::type stream=
             Stream<Allocator>::default_value) {
    m_impl.scale(v, stream);
  }

  // REFACTORING: Replace this with get_halo_width
  const IntVector &get_overlap() const {
    return get_distribution().get_overlap();
  }

  const IntVector &get_halo_width() const {
    return get_distribution().get_overlap();
  }

  int get_halo_width(int dim) const {
    return get_distribution().get_overlap(dim);
  }

  void clear_halo(int dim,
                  typename Stream<Allocator>::type stream=
                  Stream<Allocator>::default_value) {
    m_impl.clear_halo(dim, stream);
  }

  std::ostream &print(std::ostream &os) const {
    std::stringstream ss;
    ss << "(";
    ss << "shape: " << m_shape
       << ", local shape: " << get_local_shape()
       << ", local real shape: " << get_local_real_shape()
       << ", pitch: " << get_pitch()
       << ", locale: ";
    PrintLocale(ss, m_locale);
    ss << ", dist: " << m_dist
       << ", is_view?: " << m_is_view
       << ", data: " << m_data
       << ")";
    os << ss.str();
    return os;
  }

  void copyout(void *m) const {
    m_data.copyout(m);
  }

  Shape get_requested_local_shape() const {
    return m_requested_local_shape;
  }

  Shape get_requested_local_block() const {
    return m_requested_local_block;
  }

  IndexVector get_proc_index() const {
    return m_impl.get_proc_index();
  }

  IndexVector get_split_index() const {
    return m_impl.get_split_index();
  }

  bool is_split_root() const {
    return get_distribution().is_split_root(get_proc_index());
  }

 protected:
  //! Shape of the tensor
  Shape m_shape;
  // Optional shape setting; used with irregular decomposition
  Shape m_requested_local_shape;
  Shape m_requested_local_block;
  Locale m_locale;
  Distribution m_dist;

  //! Indicates whether this is a view
  bool m_is_view = false;

  Memory<Allocator> m_data;
  TensorImplType m_impl;

  void check_shape_validity() const {
    auto shape_nd = m_shape.num_dims();
    auto dist_nd = m_dist.num_dims();
    if (shape_nd != dist_nd) {
      util::MPIPrintStreamError()
          << "Mismatch between the number of tensor dimensions ("
          << shape_nd << ") and the number of distribution dimensions ("
          << dist_nd << ")";
      throw std::exception();
    }
    auto requested_nd = m_requested_local_shape.num_dims();
    if (shape_nd != requested_nd) {
      util::MPIPrintStreamError()
          << "Mismatch between the number of tensor dimensions ("
          << shape_nd << ") and the number of dimensions of requested local shape ("
          << requested_nd << ")";
      throw std::exception();
    }
  }

  void set_shape(const Shape &shape) {
    m_impl.set_shape(shape);
    check_shape_validity();
  }

  void set_distribution(const Distribution &dist) {
    m_impl.set_distribution(dist);
    check_shape_validity();
  }

  // TODO: Make these functions protected
 public:
  void set_view(const Memory<Allocator> &parent_mem) {
    m_data.alias(parent_mem);
    m_is_view = true;
  }

  void set_view(void *raw_ptr) {
    // Note: raw_ptr should not be pitched memory
    m_data.alias(raw_ptr, sizeof(DataType) * get_local_real_size(),
                 sizeof(DataType) * get_local_real_shape()[0],
                 sizeof(DataType) * get_local_real_shape()[0]);
    m_is_view = true;
  }

  void set_view(const void *raw_ptr) {
    // Note: raw_ptr should not be pitched memory
    m_data.alias(raw_ptr, sizeof(DataType) * get_local_real_size(),
                 sizeof(DataType) * get_local_real_shape()[0],
                 sizeof(DataType) * get_local_real_shape()[0]);
    m_is_view = true;
  }

  void copyin(const void *m) {
    this->m_data.copyin(m);
  }

  void set(const IndexVector &local_idx, DataType v,
           bool idx_include_halo=false) {
    index_t offset = this->get_local_offset(
        local_idx, idx_include_halo);
    this->get_buffer()[offset] = v;
    return;
  }

  /*
    Allreduces shared regions.

    Shared regions are splits that have multiple locales.
  */
  void allreduce_shared_regions() {
    this->m_impl.allreduce_shared_regions();
    return;
  }

  /*
    Allreduces along dims.
  */
  void allreduce(const std::vector<int> &dims) {
    this->m_impl.allreduce(dims);
  }
};

template <typename DataType, typename Locale, typename Allocator>
inline std::ostream &operator<<(std::ostream &os,
                                const Tensor<DataType, Locale, Allocator> &t) {
  return t.print(os);
}

#if 0
template <typename TensorType>
struct TensorTypeAddConst {
  using type = Tensor<TensorType::num_dims,
                      typename TensorType::data_type,
                      typename TensorType::locale_type,
                      typename TensorType::allocator_type>;
};

template <typename TensorType>
struct TensorTypeRemoveConst {
  using type = Tensor<TensorType::num_dims,
                      typename TensorType::data_type,
                      typename TensorType::loale_type,
                      typename TensorType::allocator_type>;
};

template <typename TensorType, bool is_const>
struct TensorTypeSetConst {
  using type = Tensor<TensorType::num_dims,
                      typename TensorType::data_type,
                      typename TensorType::locale_type,
                      typename TensorType::allocator_type>;
};
#endif
} // namespace tensor
} // namespace distconv
