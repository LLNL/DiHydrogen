#pragma once

#include "distconv/tensor/tensor.hpp"

namespace distconv {
namespace tensor {

class LocaleProcess {
 public:
  int get_size() const {
    return 1;
  }
  int get_rank() const {
    return 0;
  }
  IndexVector get_rank_idx(const Distribution &dist) const {
    return IndexVector(dist.num_dims(), 0);
  }
};

template <typename Locale>
inline std::ostream& PrintLocale(std::ostream &os, const Locale &l) {
  return os << "[" << l.get_rank() << "/" << l.get_size() << "]";
}

template <typename DataType, typename Allocator>
class TensorImpl<Tensor<DataType, LocaleProcess, Allocator>> {
  using TensorType = Tensor<DataType, LocaleProcess, Allocator>;
 public:
  TensorImpl(): m_tensor(nullptr) {}

  TensorImpl(TensorType *tensor): m_tensor(tensor) {
    if (m_tensor) update_local_real_shape();
  }

  TensorImpl(const TensorImpl<TensorType> &x):
      TensorImpl(x.m_tensor) {}

  TensorImpl(TensorType *tensor, const TensorImpl<TensorType> &x):
      TensorImpl(tensor) {}

  TensorImpl<TensorType> &operator=(const TensorImpl<TensorType> &x) {
    m_tensor = x.m_tensor;
    m_local_real_shape = 0;
    if (m_tensor) update_local_real_shape();
  }

  LocaleProcess get_sub_locale(int dim) const {
    return m_tensor->get_locale();
  }

  index_t get_local_size() const {
    return m_tensor->get_size();
  }

  Shape get_local_shape() const {
    return m_tensor->get_shape();
  }

  int allocate() {
    // When allocating tensor here, no overlap region is created.
    size_t num_elements = get_local_size();
    if (num_elements == 0) {
      std::cerr << "Can't allocate an empty tensor\n";
      return -1;
    }
    size_t ldim = get_local_shape()[0];
    return m_tensor->m_data.allocate(num_elements * sizeof(DataType),
                                     ldim * sizeof(DataType));
  }

  void nullify() {
    m_tensor->m_data.nullify();
    return;
  }

  void set_shape(const Shape &shape) {
    m_tensor->m_shape = shape;
    update_local_real_shape();
  }

  void set_distribution(const Distribution &dist) {
    m_tensor->m_dist = dist;
    update_local_real_shape();
  }

  Shape get_local_real_shape() const {
    return m_local_real_shape;
  }

  index_t get_global_index(int dim, index_t local_idx) const {
    return local_idx;
  }

  index_t get_local_index(int dim, index_t global_idx) const {
    return global_idx;
  }

  index_t get_local_offset(const IndexVector idx,
                           bool idx_include_halo=false) const {
    auto real_idx = idx;
    if (!idx_include_halo) {
      real_idx = real_idx + m_tensor->get_distribution().get_overlap();
    }
    return get_offset(
        real_idx, get_local_real_shape(), m_tensor->get_pitch());
  }

  void exchange_halo(int dim) {
    // No exchange is needed as this tensor is not distributed
    return;
  }

  index_t get_dimension_rank_offset(int dim, int rank) const {
    return 0;
  }

 protected:
  void update_local_real_shape() {
    const auto &dist = m_tensor->get_distribution();
    m_local_real_shape = Shape(m_tensor->get_num_dims(), 0);
    for (int i = 0; i < m_tensor->get_num_dims(); ++i) {
      m_local_real_shape[i] = get_local_shape()[i] +
          dist.get_overlap(i) * 2;
    }
  }
  TensorType *m_tensor;
  Shape m_local_real_shape;
};

template <typename DataType, typename Allocator>
inline int View(Tensor<DataType, LocaleProcess, Allocator> &t_proc,
                DataType *raw_ptr) {
  t_proc.set_view(raw_ptr);
  return 0;
}

template <typename DataType, typename Allocator>
inline int View(Tensor<DataType, LocaleProcess, Allocator> &t_proc,
                const DataType *raw_ptr) {
  t_proc.set_view(raw_ptr);
  return 0;
}

} // namespace tensor
} // namespace distconv
