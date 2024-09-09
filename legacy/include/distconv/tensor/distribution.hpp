#pragma once

#include "distconv/base.hpp"
#include "distconv/tensor/tensor_base.hpp"
#include "distconv/util/util.hpp"
#include "distconv/vector.hpp"

#include <iostream>
#include <sstream>

namespace distconv
{
namespace tensor
{

class Distribution
{
private:
  Shape m_locale_shape;
  Shape m_split_shape;
  IntVector m_overlap;
  // Block size when cyclic distribution is used
  Shape m_block_size;

public:
  Distribution(const Shape& locale_shape,
               const Shape& split_shape,
               const IntVector& overlap,
               const Shape& block_size)
    : m_locale_shape(locale_shape),
      m_split_shape(split_shape),
      m_overlap(overlap),
      m_block_size(block_size)
  {
    sanity_check_shapes();
    fixup_overlap();
  }

private:
  Distribution(const Shape& locale_shape,
               const Shape& split_shape,
               const IntVector& overlap)
    : Distribution(
        locale_shape, split_shape, overlap, Shape(locale_shape.num_dims(), 0))
  {}

  Distribution(const Shape& locale_shape, const Shape& split_shape)
    : Distribution(
        locale_shape, split_shape, IntVector(locale_shape.num_dims(), 0))
  {}

  Distribution(const Shape& locale_shape)
    : Distribution(locale_shape, locale_shape)
  {}

public:
  Distribution() = default;
  Distribution(int num_dims) : Distribution(Shape(num_dims, 1)) {}

  static Distribution make_localized_distribution(int num_dims)
  {
    return Distribution(num_dims);
  }

  static Distribution make_distribution(const Shape& locale_shape)
  {
    return Distribution(locale_shape);
  }

  static Distribution make_distribution(std::initializer_list<int> locale_shape)
  {
    return Distribution(Shape(locale_shape));
  }

  static Distribution make_shared_distribution(const Shape& locale_shape)
  {
    int nd = locale_shape.num_dims();
    return Distribution(locale_shape, Shape(nd, 1));
  }

  static Distribution make_shared_distribution(const Shape& locale_shape,
                                               const Shape& split_shape)
  {
    return Distribution(locale_shape, split_shape);
  }

  static Distribution
  make_shared_distribution(std::initializer_list<int> locale_shape,
                           std::initializer_list<int> split_shape)
  {
    return make_shared_distribution(Shape(locale_shape), Shape(split_shape));
  }

  static Distribution make_overlapped_distribution(const Shape& locale_shape,
                                                   const IntVector& overlap)
  {
    return Distribution(locale_shape, locale_shape, overlap);
  }

  static Distribution
  make_overlapped_distribution(std::initializer_list<int> locale_shape,
                               std::initializer_list<int> overlap)
  {
    return make_overlapped_distribution(Shape(locale_shape),
                                        IntVector(overlap));
  }

  int num_dims() const { return m_locale_shape.num_dims(); }

  bool operator==(const Distribution& d) const
  {
    return m_locale_shape == d.m_locale_shape
           && m_split_shape == d.m_split_shape && m_block_size == d.m_block_size
           && m_overlap == d.m_overlap;
  }

  bool operator!=(const Distribution& d) const { return !operator==(d); }

  Distribution get_non_overlapped_distribution() const
  {
    Distribution non_overlapped_dist(*this);
    non_overlapped_dist.clear_overlap();
    return non_overlapped_dist;
  }

  const Shape& get_locale_shape() const { return m_locale_shape; }

  index_t get_locale_shape(int dim) const { return m_locale_shape[dim]; }

  void set_locale_shape(const Shape& locale_shape)
  {
    m_locale_shape = locale_shape;
    sanity_check_shapes();
  }

  const Shape& get_split_shape() const { return m_split_shape; }

  void set_split_shape(const Shape& split_shape)
  {
    m_split_shape = split_shape;
    sanity_check_shapes();
  }

  void sanity_check_shapes() const
  {
    auto nd = num_dims();
    assert_eq(m_split_shape.num_dims(), nd);
    assert_eq(m_block_size.num_dims(), nd);
    assert_eq(m_overlap.length(), nd);
    for (int i = 0; i < nd; ++i)
    {
      if (m_locale_shape[i] % m_split_shape[i] != 0)
      {
        util::PrintStreamError()
          << "m_locale_shape[" << i << "] not divisible by m_split_shape[" << i
          << "]: " << m_locale_shape[i] << " % " << m_split_shape[i];
        assert0(m_locale_shape[i] % m_split_shape[i]);
      }
    }
  }

  Shape get_num_ranks_per_split() const
  {
    return get_locale_shape() / get_split_shape();
  }

  index_t get_num_ranks_per_split(int dim) const
  {
    return get_num_ranks_per_split()[dim];
  }

  const IntVector& get_overlap() const { return m_overlap; }

  int get_overlap(int dim) const { return get_overlap()[dim]; }

  const Shape& get_block_size() const { return m_block_size; }

  index_t get_block_size(int dim) const { return m_block_size[dim]; }

  void set_overlap(int dim, int o)
  {
    m_overlap[dim] = o;
    return;
  }

  void set_overlap(const IntVector& overlap)
  {
    m_overlap = overlap;
    return;
  }

  void clear_overlap() { m_overlap = 0; }

  void copy_overlap(const Distribution& d) { m_overlap = d.m_overlap; }

  bool is_distributed(int dim) const { return get_split_shape()[dim] > 1; }

  bool is_distributed() const
  {
    bool distributed = false;
    for (int i = 0; i < num_dims(); ++i)
    {
      distributed = distributed || is_distributed(i);
    }
    return distributed;
  }

  bool is_shared(int dim) const { return get_num_ranks_per_split(dim) > 1; }

  bool is_shared() const
  {
    bool shared = false;
    for (int i = 0; i < num_dims(); ++i)
    {
      shared = shared || is_shared(i);
    }
    return shared;
  }

  bool is_split_root(int dim, int rank) const
  {
    return rank % get_num_ranks_per_split(dim) == 0;
  }

  template <int ND>
  bool is_split_root(const Array<ND>& rank) const
  {
    for (int i = 0; i < num_dims(); ++i)
    {
      if (!is_split_root(i, rank[i]))
        return false;
    }
    return true;
  }

  bool is_split_root(const IntVector& rank) const
  {
    for (int i = 0; i < num_dims(); ++i)
    {
      if (!is_split_root(i, rank[i]))
        return false;
    }
    return true;
  }

  bool is_multi_dimensional() const
  {
    int num_partitioned_dims = 0;
    for (int i = 0; i < num_dims(); ++i)
    {
      if (m_locale_shape[i] > 1)
      {
        ++num_partitioned_dims;
      }
    }
    return num_partitioned_dims > 1;
  }

  // disables overlap if the dimension is not distributed
  void fixup_overlap()
  {
    for (int i = 0; i < num_dims(); ++i)
    {
      if (!is_distributed(i))
      {
        set_overlap(i, 0);
      }
    }
  }

  std::ostream& print(std::ostream& os) const
  {
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < num_dims(); ++i)
    {
      if (i != 0)
      {
        ss << ", ";
      }
      ss << m_split_shape[i] << "/" << m_locale_shape[i] << ":" << m_overlap[i];
    }
    ss << ")";
    return os << ss.str();
  }
};

inline std::ostream& operator<<(std::ostream& os, const Distribution& d)
{
  return d.print(os);
}

} // namespace tensor
} // namespace distconv
