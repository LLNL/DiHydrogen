#pragma once

#include "distconv/util/util.hpp"
#include "distconv/base.hpp"
#include "distconv/vector.hpp"
#include <initializer_list>
#include <assert.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <iterator>
#include <numeric>

#if defined(__CUDACC__) || __HIP__
#define TENSOR_FUNC_DECL __host__ __device__
#else
#define TENSOR_FUNC_DECL
#endif

namespace distconv {
namespace tensor {

// TODO: Change ND to SIZE
template <int ND, typename DataType=index_t>
struct Array {
 public:
  static constexpr int num_dims = ND;
  using data_type = DataType;
  TENSOR_FUNC_DECL
  Array() {
    for (int i = 0; i < ND; ++i) {
      dims[i] = 0;
    }
  }
  TENSOR_FUNC_DECL
  Array(DataType d) {
    for (int i = 0; i < ND; ++i) {
      dims[i] = d;
    }
  }
  template <typename T>
  TENSOR_FUNC_DECL Array(std::initializer_list<T> d) {
    auto it = d.begin();
    for (int i = 0; i < ND; ++i) {
      dims[i] = *it;
      ++it;
    }
  }
  template <typename T>
  Array(const std::vector<T> &d) {
    assert_always(ND == d.size());
    auto it = d.begin();
    for (int i = 0; i < ND; ++i) {
      dims[i] = *it;
      ++it;
    }
  }

  template <typename T>
  Array(const Vector<T> &d) {
    assert_always(ND == d.length());
    auto it = d.begin();
    for (int i = 0; i < ND; ++i) {
      dims[i] = *it;
      ++it;
    }
  }

  TENSOR_FUNC_DECL DataType operator[](int rank) const {
    rank = rank < 0 ? ND + rank : rank;
    return dims[rank];
  }
  TENSOR_FUNC_DECL DataType& operator[](int rank) {
    rank = rank < 0 ? ND + rank : rank;
    return dims[rank];
  }
  TENSOR_FUNC_DECL DataType front() const {
    return operator[](0);
  }
  TENSOR_FUNC_DECL DataType& front() {
    return operator[](0);
  }
  TENSOR_FUNC_DECL DataType back() const {
    return operator[](-1);
  }
  TENSOR_FUNC_DECL DataType& back() {
    return operator[](-1);
  }
  TENSOR_FUNC_DECL bool operator==(const Array<ND, DataType> &s) const {
    for (int i = 0; i < ND; ++i) {
      if (dims[i] != s.dims[i]) {
        return false;
      }
    }
    return true;
  }
  TENSOR_FUNC_DECL bool operator!=(const Array<ND, DataType> &s) const {
    return !(*this == s);
  }

  TENSOR_FUNC_DECL DataType get_size() const {
    return reduce_prod();
  }

  TENSOR_FUNC_DECL DataType reduce_prod() const {
    DataType ne = 1;
    for (int i = 0; i < ND; ++i) {
      ne *= dims[i];
    }
    return ne;
  }

  TENSOR_FUNC_DECL DataType reduce_sum() const {
    DataType ne = 0;
    for (int i = 0; i < ND; ++i) {
      ne += dims[i];
    }
    return ne;
  }

  TENSOR_FUNC_DECL bool is_empty() const {
    return get_size() == 0;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator+(
      const Array<ND, DataTypeX> &x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] + x[i];
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator-(
      const Array<ND, DataTypeX> &x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] - x[i];
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator*(
      const Array<ND, DataTypeX> &x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] * x[i];
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator/(
      const Array<ND, DataTypeX> &x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] / x[i];
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator+(
      DataTypeX x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] + x;
    }
    return sum;
  }

  template <typename DataTypeX>
  Array<ND, DataType> operator+(
      const Vector<DataTypeX> &x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] + x[i];
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator-(
      DataTypeX x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] - x;
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator*(
      DataTypeX x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] * x;
    }
    return sum;
  }

  template <typename DataTypeX>
  TENSOR_FUNC_DECL Array<ND, DataType> operator/(
      DataTypeX x) const {
    Array<ND, DataType> sum;
    for (int i = 0; i < ND; ++i) {
      sum[i] = dims[i] / x;
    }
    return sum;
  }

  IndexVector get_vector() const {
    IndexVector v;
    for (int i = 0; i < ND; ++i) {
      v.push_back(dims[i]);
    }
    return v;
  }

 private:
  DataType dims[ND];
};

template <int ND, typename DataType>
inline std::ostream &operator<<(std::ostream &os, const Array<ND, DataType> &s) {
  std::stringstream ss;
  ss << "(";
  for (int i = 0; i < ND; ++i) {
    if (i != 0) {
      ss << ", ";
    }
    ss << s[i];
  }
  ss << ")";
  return os << ss.str();
}

template <int ND, typename DataType>
inline Array<ND, DataType> MakeArray(const std::vector<DataType> &v) {
  assert_always(ND == v.size());
  Array<ND, DataType> a;
  for (int i = 0; i < ND; ++i) {
    a[i] = v[i];
  }
  return a;
}

template <int ND, typename DataType=index_t>
class ArrayTraverser {
 public:
  static constexpr int num_dims = ND;
  using data_type = DataType;
  ArrayTraverser(const Array<ND, DataType> &array):
      m_array(array), m_index(0) {}

  static ArrayTraverser<ND, DataType> begin(const Array<ND, DataType> &a) {
    auto t = ArrayTraverser<ND, DataType>(a);
    if (a.get_size() == 0) {
      t.set_end();
    }
    return t;
  }

  static ArrayTraverser<ND, DataType> end(const Array<ND, DataType> &a) {
    auto it = ArrayTraverser<ND, DataType>(a);
    it.set_end();
    return it;
  }

  Array<ND, DataType> operator*() const {
    return m_index;
  }

  Array<ND> operator++() {
    Array<ND> cur = m_index;
    for (int i = 0; i < ND; ++i) {
      if (m_index[i] < m_array[i]) {
        ++m_index[i];
        if (m_index[i] == m_array[i]) {
          m_index[i] = 0;
        } else {
          return cur;
        }
      }
    }
    Array<ND, DataType> zero(0);
    assert_always(m_index == zero);
    m_index = m_array;
    return cur;
  }

  bool operator==(const ArrayTraverser<ND, DataType> &it) const {
    return m_index == it.m_index && m_array == it.m_array;
  }

  bool operator!=(const ArrayTraverser<ND, DataType> &it) const {
    return !(*this == it);
  }

  void set_end() {
    m_index = m_array;
  }

 protected:
  const Array<ND, DataType> &m_array;
  Array<ND, DataType> m_index;
};

template <int ND, typename IndexType>
TENSOR_FUNC_DECL
index_t get_offset(const Array<ND, IndexType> &index,
                   const Array<ND, IndexType> &shape) {
  return get_offset(index, shape, shape[0]);
}

#if 0
template <int ND, typename IndexType>
TENSOR_FUNC_DECL
index_t get_offset(const Array<ND, IndexType> &index,
                   const Array<ND, IndexType> &shape,
                   IndexType pitch) {
  IndexType offset = 0;
  IndexType stride = 1;
  assert_always(pitch != 0);
  for (int i = 0; i < ND; ++i) {
    offset += stride * (index[i] + overlap[i]);
    if (i == 0) {
      //stride *= pitch ? pitch : shape[0];
      stride *= pitch;
    } else {
      stride *= shape[i];
    }
  }
  return offset;
}
#endif
template <int ND, typename IndexType>
TENSOR_FUNC_DECL
index_t get_offset(const Array<ND, IndexType> &index,
                   const Array<ND, IndexType> &shape,
                   IndexType pitch) {
  IndexType offset = 0;
  IndexType stride = 1;
  assert(pitch != 0);
  for (int i = 0; i < ND; ++i) {
    offset += stride * index[i];
    if (i == 0) {
      stride *= pitch;
    } else {
      stride *= shape[i];
    }
  }
  return offset;
}

template <int ND, typename IndexType>
TENSOR_FUNC_DECL
Array<ND, IndexType> get_index(IndexType offset,
                               const Array<ND, IndexType> &shape) {
  Array<ND> index;
  for (int i = 0; i < ND; ++i) {
    index[i] = offset % shape[i];
    offset /= shape[i];
  }
  return index;
}

class Shape: public Vector<index_t> {
 public:
  using Vector::Vector;

  Shape() = default;

  Shape(const Vector<index_t> &v):
      Vector(v) {}

  template <int ND>
  Shape(const Array<ND, index_t> &a):
      Vector(ND) {
    for (int i = 0; i < ND; ++i) {
      (*this)[i] = a[i];
    }
  }

  using Vector::operator=;

  int num_dims() const {
    return length();
  }

  size_t size() const {
    return reduce_prod();
  }

  // For compatibility with Array
  size_t get_size() const {
    return size();
  }

  bool is_empty() const {
    return reduce_prod() == 0;
  }

  IndexVector get_index(index_t offset) const {
    IndexVector index(num_dims());
    for (int i = 0; i < num_dims(); ++i) {
      index[i] = offset % (*this)[i];
      offset /= (*this)[i];
    }
    return index;
  }

  class IndexIterator {
   public:
    IndexIterator(const Shape &shape):
        m_shape(shape), m_index(shape.num_dims(), 0) {
      if (shape.size() == 0) {
        set_end();
      }
    }

    IndexVector operator*() const {
      return m_index;
    }

    IndexVector operator++() {
      auto cur = m_index;
      for (int i = 0; i < m_shape.length(); ++i) {
        if (m_index[i] < m_shape[i]) {
          ++m_index[i];
          if (m_index[i] == m_shape[i]) {
            m_index[i] = 0;
          } else {
            return cur;
          }
        }
      }
      // have traversed all indices. sanity check.
      assert0(m_index);
      m_index = m_shape;
      return cur;
    }

    bool operator==(const IndexIterator &it) const {
      return m_index == it.m_index && m_shape == it.m_shape;
    }

    bool operator!=(const IndexIterator &it) const {
      return !(*this == it);
    }

    void set_end() {
      m_index = m_shape;
    }

   protected:
    const IndexVector m_shape;
    IndexVector m_index;
  };

  IndexIterator index_begin() const {
    return IndexIterator(*this);
  }

  IndexIterator index_end() const {
    auto it = IndexIterator(*this);
    it.set_end();
    return it;
  }
};

template <int ND, typename IndexType>
index_t get_offset(const Array<ND, IndexType> &index,
                   const Shape &shape) {
  return get_offset(index, Array<ND, IndexType>(shape));
}

template <int ND, typename IndexType>
index_t get_offset(const Array<ND, IndexType> &index,
                   const Shape &shape,
                   IndexType pitch) {
  return get_offset(index, Array<ND, IndexType>(shape), pitch);
}

// REFACTORING: duplicate of the above functions
inline index_t get_offset(const IndexVector &index,
                          const Shape &shape,
                          index_t pitch) {
  index_t offset = 0;
  index_t stride = 1;
  assert(pitch != 0);
  for (int i = 0; i < index.length(); ++i) {
    offset += stride * index[i];
    if (i == 0) {
      stride *= pitch;
    } else {
      stride *= shape[i];
    }
  }
  return offset;
}

inline index_t get_offset(const IndexVector &index,
                          const Shape &shape) {
  return get_offset(index, shape, shape[0]);
}

class Region {
  using IndexType = IndexVector;
  using ExtentType = Shape;
 public:
  Region() = default;
  Region(const IndexType &offset, const ExtentType &extent):
      m_offset(offset), m_extent(extent) {
    assert_eq(m_offset.length(), m_extent.num_dims());
  }
  const IndexType &get_offset() const {
    return m_offset;
  }
  const ExtentType &get_extent() const {
    return m_extent;
  }

  index_t get_size() const {
    return get_extent().get_size();
  }

  bool is_empty() const {
    return get_size() == 0;
  }

  int num_dims() const {
    return m_extent.num_dims();
  }

  Region intersect(const Region &x) const {
    assert_eq(num_dims(), x.num_dims());
    Region r(IndexType(num_dims(), 0), ExtentType(num_dims(), 0));
    for (int i = 0; i < num_dims(); ++i) {
      index_t offset = std::max(m_offset[i], x.m_offset[i]);
      index_t rh_end = std::min(
          m_offset[i] + m_extent[i],
          x.m_offset[i] + x.m_extent[i]);
      if (offset >= rh_end) {
        // disjoint
        r.m_offset = 0;
        r.m_extent = 0;
        break;
      }
      index_t extent = rh_end - offset;
      r.m_offset[i] = offset;
      r.m_extent[i] = extent;
    }
    return r;
  }

 protected:
  IndexVector m_offset;
  Shape m_extent;
};

inline std::ostream &operator<<(std::ostream &os, const Region &r) {
  std::stringstream ss;
  ss << "(offset: " << r.get_offset()
     << ", extent: " << r.get_extent() << ")";
  return os << ss.str();
}

template <int ND>
Array<ND> get_strides(const Array<ND> &logical_shape,
                      const Array<ND> &overlap,
                      index_t pitch) {
  Array<ND> real_shape = logical_shape + overlap * 2;
  Array<ND> strides(1);
  for (int i = 1; i < ND; ++i) {
    if (i == 1) {
      strides[i] = pitch;
    } else {
      strides[i] = strides[i-1] * real_shape[i-1];
    }
  }
  return strides;
}

template <int ND>
Array<ND> get_strides(const Array<ND> &logical_shape,
                      const IntVector &overlap,
                      index_t pitch) {
  return get_strides(logical_shape, Array<ND>(overlap), pitch);
}


inline IndexVector get_strides(const Shape &logical_shape,
                               const IntVector &overlap,
                               index_t pitch) {
  Shape real_shape = logical_shape + overlap * 2;
  real_shape[0] = pitch;
  IntVector strides;
  strides.push_back(1);
  std::partial_sum(real_shape.begin(), real_shape.end() - 1,
                   std::back_inserter(strides),
                   std::multiplies<index_t>());
  return strides;
}

} // namespace tensor
} // namespace distconv
