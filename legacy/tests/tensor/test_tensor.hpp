#pragma once

#include "distconv/tensor/tensor.hpp"
#include "distconv/util/util.hpp"
#include <iostream>

using namespace distconv;
using namespace distconv::tensor;

template <int ND>
index_t get_linearlized_offset(const Array<ND> &offset,
                               const Array<ND> &dim) {
  index_t x = 0;
  index_t d = 1;
  for (int i = 0; i < ND; ++i) {
    x += offset[i] * d;
    d *= dim[i];
  }
  return x;
}

index_t get_linearlized_offset(const IndexVector &offset,
                               const Shape &dim) {
  index_t x = 0;
  index_t d = 1;
  for (int i = 0; i < offset.length(); ++i) {
    x += offset[i] * d;
    d *= dim[i];
  }
  return x;
}

template <typename Locale>
inline Locale get_locale() {
  Locale loc;
  return loc;
}

template <typename TensorType>
inline TensorType get_tensor(typename TensorType::locale_type &loc,
                             const Distribution &dist) {
  TensorType tensor(loc, dist);
  return tensor;
}

template <typename TensorType>
inline TensorType get_tensor(const Shape &shape,
                             typename TensorType::locale_type &loc,
                             const Distribution &dist) {
  TensorType tensor(shape, loc, dist);
  return tensor;
}

template <typename TensorType>
inline int test_alloc(const Shape &shape,
                      const Distribution &dist) {
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  assert_always(!t.is_view());
  util::PrintStreamDebug() << "Shape: " << t.get_shape();
  assert_always(t.is_null());
  util::PrintStreamDebug() << "Distribution: " << t.get_distribution();
  assert0(t.allocate());
  assert_always(t.get_local_size() == 0 || t.is_non_null());
  TensorType t2 = t;
  assert0(t.nullify());
  assert_always(t.is_null());
  assert_always(t2.get_local_size() == 0 || t2.is_non_null());

  return 0;
}

template <typename TensorType>
inline int test_data_access(const Shape &shape,
                            const Distribution &dist) {
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  util::PrintStreamDebug() << "Shape: " << t.get_shape() << std::endl;
  util::PrintStreamDebug() << "Distribution: " << t.get_distribution()
                           << std::endl;

  assert0(t.allocate());

  if (t.get_local_size()) {
    auto local_shape = t.get_local_shape();
    index_t base_offset = t.get_local_offset();
    auto *buf = t.get_buffer();
    size_t ldim = t.get_pitch();
    for (index_t i = 0; i < local_shape[2]; ++i) {
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t k = 0; k < local_shape[0]; ++k) {
          buf[base_offset + k + ldim * j + ldim * local_shape[1] * i]
              = t.get_global_index(0, k)
              + t.get_global_index(1, j) * t.get_shape()[0]
              + t.get_global_index(2, i) * t.get_shape()[0] * t.get_shape()[1];
        }
      }
    }
    for (index_t i = 0; i < local_shape[2]; ++i) {
      std::cout << "[][][" << i << "]\n";
      for (index_t j = 0; j < local_shape[1]; ++j) {
        for (index_t k = 0; k < local_shape[0]; ++k) {
          IndexVector idx({k, j, i});
          std::cout << buf[t.get_local_offset(idx)]
                    << "@" << t.get_local_offset(idx) << " ";
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }
  return 0;
}

template <typename TensorType>
inline int test_data_access4(const Shape &shape,
                             const Distribution &dist) {
  using LocaleType = typename TensorType::locale_type;
  LocaleType loc = get_locale<LocaleType>();
  TensorType t = get_tensor<TensorType>(shape, loc, dist);
  util::PrintStreamDebug() << "Shape: " << t.get_shape() << std::endl;
  util::PrintStreamDebug() << "Distribution: " << t.get_distribution()
                           << std::endl;

  assert0(t.allocate());

  if (t.get_local_size()) {
    Array<4> local_shape = t.get_local_shape();
    index_t base_offset = t.get_local_offset();
    auto *buf = t.get_buffer();
    size_t ldim = t.get_pitch();
    for (index_t i = 0; i < local_shape[3]; ++i) {
      for (index_t j = 0; j < local_shape[2]; ++j) {
        for (index_t k = 0; k < local_shape[1]; ++k) {
          for (index_t l = 0; l < local_shape[0]; ++l) {
            buf[base_offset + l + ldim * k + ldim * local_shape[1] * j
                + ldim * local_shape[1] * local_shape[2] * i]
                = t.get_global_index(0, l)
                + t.get_global_index(1, k) * t.get_shape()[0]
                + t.get_global_index(2, j) * t.get_shape()[0] * t.get_shape()[1]
                + t.get_global_index(3, i) * t.get_shape()[0] * t.get_shape()[1] * t.get_shape()[2];
          }
        }
      }
    }
    for (index_t i = 0; i < local_shape[3]; ++i) {
      for (index_t j = 0; j < local_shape[2]; ++j) {
        for (index_t k = 0; k < local_shape[1]; ++k) {
          for (index_t l = 0; l < local_shape[0]; ++l) {
            IndexVector idx({l, k, j, i});
            std::cout << buf[t.get_local_offset(idx)]
                      << "@" << t.get_local_offset(idx) << " ";
          }
        }
      }
    }
    std::cout << std::endl;
  }
  return 0;
}
