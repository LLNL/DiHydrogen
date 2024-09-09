#pragma once

#include "distconv/base.hpp"
#include "distconv/util/util.hpp"

#include <functional>
#include <initializer_list>
#include <vector>

namespace distconv
{

template <typename DataType>
class Vector
{
private:
  using container_type = std::vector<DataType>;
  container_type m_data;

public:
  using data_type = DataType;
  using value_type = data_type;  // for compatibility with std::vector
  using iterator = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;
  using reverse_iterator = typename container_type::reverse_iterator;
  using const_reverse_iterator =
    typename container_type::const_reverse_iterator;
  using reference = typename container_type::reference;
  using const_reference = typename container_type::const_reference;

  /**
     Constructs an empty vector.
   */
  explicit Vector() = default;

  /**
     Constructs a vector of a given dimension.

     @param dim The vector dimension.
   */
  explicit Vector(int dim) : m_data(dim) {}

  /**
     Constructs a vector of a given dimension with all elements
     initilized with a scalar value.

     @param dim The vector dimension.
     @param x The initial value of all elements.
   */
  explicit Vector(int dim, const DataType& x) : m_data(dim, x) {}

  /**
     Constructs a vector by copying another vector of possibly
     different type. Eeach element of type T is assigned to this
     vector of type DataType with implicit type casting if
     necessary.Copy constructor.

     @param v A vector of the same type.
   */
  template <typename T>
  Vector(const Vector<T>& v)
  {
    for (const auto& x : v)
    {
      m_data.push_back(x);
    }
  }

  /**
     Constructs a vector by copying a std::vector of type T. Each
     element of type T is assigned to this vector of type DataType
     with implicit type casting if necessary.

     @param v A std::vector to copy.
   */
  template <typename T>
  explicit Vector(const std::vector<T>& v)
  {
    for (const auto& x : v)
    {
      m_data.push_back(x);
    }
  }

  /**
     Constructs a vector by copying an initializer list of type
     T. Each element of type T is assigned to this vector of type
     DataType with implicit type casting if necessary.

     @param l An initializer list.
   */
  template <typename T>
  explicit Vector(std::initializer_list<T> l)
  {
    for (const auto& x : l)
    {
      m_data.push_back(x);
    }
  }

  /**
     Constructs a vector by copying the contents of a range.

     @param first An iterator to the beginning of a range.
     @param last An iterator to the end of a range.
   */
  template <typename InputIt>
  explicit Vector(InputIt first, InputIt last) : m_data(first, last)
  {}

  virtual ~Vector() = default;

  /**
     Returns a reference to the element at specified location
     idx. Similar to Python, negative offsets are allowed to index
     starting from the last element.

     @param idx Offset.
     @return Reference to the element at the offset.
   */
  reference operator[](int idx)
  {
    idx = idx < 0 ? length() + idx : idx;
    assert_always(idx >= 0);
    assert_always(length() > 0);
    assert_always(idx < length());
    return m_data.at(idx);
  }

  /**
     Returns a const reference to the element at specified location
     idx. Similar to Python, negative offsets are allowed to index
     starting from the last element.

     @param idx Offset.
     @return Reference to the element at the offset.
   */
  const_reference operator[](int idx) const
  {
    idx = idx < 0 ? length() + idx : idx;
    assert_always(idx >= 0);
    assert_always(length() > 0);
    assert_always(idx < length());
    return m_data.at(idx);
  }

  /**
     Assigns a copy of another vector to this vector.

     @param x A vector to copy.
     @return *this.
   */
  Vector operator=(const Vector& v)
  {
    m_data = v.m_data;
    return *this;
  }

  /**
     Assigns a value of type DataType to all vector elements.

     @param x A scalar value to assign.
     @return *this.
   */
  Vector operator=(const DataType& x)
  {
    for (auto& i : m_data)
    {
      i = x;
    }
    return *this;
  }

  /**
     Appends a value to the end of the vector.

     @param x Value to append.
   */
  void push_back(const DataType& x) { m_data.push_back(x); }

  /**
     This is equivalent to size() of std::vector, however, it can be
     confusing if it were named as such when used to represent a shape
     of a tensor, which is one of the main use cases of this vector
     class. The size of a shape would imply the number of elements the
     shape can hold, thus the product of all elements in the vector
     should be returned, rather than the lenght of the vector.

     @return The dimension of the vector.
   */
  int length() const { return m_data.size(); }

  /**
     Computes the product of all elements in the vector.

     @return The computed product.
   */
  DataType reduce_prod() const
  {
    return reduce(std::multiplies<DataType>(), DataType(1));
  }

  /**
     Computes the sum of all elements in the vector.

     @return The computed sum.
   */
  DataType reduce_sum() const
  {
    return reduce(std::plus<DataType>(), DataType(0));
  }

  /**
     @return A pointer to the underlying element storage.
   */
  DataType* data() { return m_data.data(); }

  /**
     @return A pointer to the underlying element storage.
   */
  const DataType* data() const { return m_data.data(); }

  template <typename T>
  std::vector<T> get_vector() const
  {
    std::vector<T> v;
    std::copy(begin(), end(), std::back_inserter(v));
    return v;
  }

  /**
     Maps a unary function element-wise that takes one DataType
     parameter and returns a DataType value.

     @param m A unary mapper function.
     @return The result vector.
   */
  template <typename Mapper>
  Vector map(Mapper m) const
  {
    Vector r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i]);
    }
    return r;
  }

  /**
     Maps a unary function element-wise that takes one DataType
     parameter and returns a value of a given type.

     @param m A unary mapper function.
     @return The result vector.
   */
  template <typename Mapper, typename MapType>
  Vector<MapType> map(Mapper m) const
  {
    Vector<MapType> r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i]);
    }
    return r;
  }

  /**
     Maps a binary function element-wise that takes two DataType
     parameters and returns a DataType value.

     @param m A binary mapper function.
     @param v An RHS operand vector.
     @return The result vector.
   */
  template <typename Mapper>
  Vector map(Mapper m, const Vector& v) const
  {
    assert_eq(length(), v.length());
    Vector r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i], v[i]);
    }
    return r;
  }

  /**
     Maps a binary function element-wise that takes two DataType
     parameters and returns a value of a given type.

     @param m A binary mapper function.
     @param v An RHS operand vector.
     @return The result vector.
   */
  template <typename Mapper, typename MapType>
  Vector<MapType> map(Mapper m, const Vector& v) const
  {
    assert_eq(length(), v.length());
    Vector<MapType> r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i], v[i]);
    }
    return r;
  }

  /**
     Maps a binary function element-wise that takes two DataType
     parameters and returns a DataType value.

     @param m A binary mapper function.
     @param x A scalar value used as the RHS operand.
     @return The result vector.
   */
  template <typename Mapper>
  Vector map(Mapper m, const DataType& x) const
  {
    Vector r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i], x);
    }
    return r;
  }

  /**
     Maps a binary function element-wise that takes two DataType
     parameters and returns a value of a given type.

     @param m A binary mapper function.
     @param x A scalar value used as the RHS operand.
     @return The result vector.
   */
  template <typename Mapper, typename MapType>
  Vector<MapType> map(Mapper m, const DataType& x) const
  {
    Vector<MapType> r(length());
    for (int i = 0; i < length(); ++i)
    {
      r[i] = m((*this)[i], x);
    }
    return r;
  }

  /**
     Reduces the vector to a single DataType value by a binary
     reduction function.

     @param r A reduction function.
     @param init A scalar value used as the initial value.
     @return The reduced value.
   */
  template <typename Reducer>
  DataType reduce(Reducer r, const DataType& init) const
  {
    DataType result = init;
    for (int i = 0; i < length(); ++i)
    {
      result = r(result, (*this)[i]);
    }
    return result;
  }

  /**
     Reduces the result vector of mapping a unary function.

     This is equivalent to return map<Mapper, MapType>(m, v).reduce(r,
     init) but slightly more efficient as mapping and reduction are
     fused.

     @param m A unary mapper function.
     @param r A reduction function.
     @param init A scalar value used as the initial value.
     @return The reduced value.
   */
  template <typename Mapper, typename Reducer, typename MapType>
  MapType map_reduce(Mapper m, Reducer r, const MapType& init) const
  {
    MapType result = init;
    for (int i = 0; i < length(); ++i)
    {
      result = r(result, m((*this)[i]));
    }
    return result;
  }

  /**
     Reduces the result vector of mapping a binary function.

     This is equivalent to return map<Mapper, MapType>(m, v).reduce(r,
     init) but slightly more efficient as mapping and reduction are
     fused.

     @param m A binary mapper function.
     @param v A vector for the RHS operand of the binary mapper.
     @param r A reduction function.
     @param init A scalar value used as the initial value.
     @return The reduced value.
   */
  template <typename Mapper, typename Reducer, typename MapType>
  MapType
  map_reduce(Mapper m, const Vector& v, Reducer r, const MapType& init) const
  {
    assert_eq(length(), v.length());
    DataType result = init;
    for (int i = 0; i < length(); ++i)
    {
      result = r(result, m((*this)[i], v[i]));
    }
    return result;
  }

  /**
     Computes element-wise summation with a vector.

     @param v A vector of the same type.
     @return The computed vector.
   */
  Vector<DataType> operator+(const Vector& v) const
  {
    return map(std::plus<DataType>(), v);
  }

  /**
     Computes an element-wise subtraction with a vector.

     @param v A vector of the same type.
     @return The computed vector.
   */
  Vector operator-(const Vector& v) const
  {
    return map(std::minus<DataType>(), v);
  }

  /**
     Computes an element-wise multiply with a vector.

     @param v A vector of the same type.
     @return The computed vector.
   */
  Vector operator*(const Vector& v) const
  {
    return map(std::multiplies<DataType>(), v);
  }

  /**
     Computes an element-wise division with a vector.

     @param v A vector of the same type.
     @return The computed vector.
   */
  Vector operator/(const Vector& v) const
  {
    return map(std::divides<DataType>(), v);
  }

  /**
     Checks if all elements are equal.

     @param v A vector to compare with.
     @return True if they are equal.
   */
  bool operator==(const Vector& v) const
  {
    return map_reduce(
      std::equal_to<DataType>(), v, std::logical_and<bool>(), true);
  }

  /**
     Checks if any elements are not equal.

     This is equivalent to !(this == v).

     @param v A vector to compare with.
     @return True if there is any mismatch.
   */
  bool operator!=(const Vector& v) const { return !operator==(v); }

  /**
     Checks if all elements are equal.

     @param v A scalar value to compare with.
     @return True if they are equal.
   */
  bool operator==(const DataType& x) const
  {
    return map<std::equal_to<DataType>, bool>(std::equal_to<DataType>(), x)
      .reduce(std::logical_and<bool>(), true);
  }

  /**
     Checks if any elements are not equal.

     This is equivalent to !(this == v).

     @param v A scalar value to compare with.
     @return True if there is any mismatch.
   */
  bool operator!=(const DataType& x) const { return !operator==(x); }

  /**
     Adds a vector element-wise.

     @param v A vector to add.
     @return *this.
   */
  Vector operator+=(const Vector& v)
  {
    return (*this) = map(std::plus<DataType>(), v);
  }

  /**
     Computes element-wise multiplication with a scalar value.

     @param x A scalar multiplier value.
     @return The result vector.
   */
  Vector operator*(const DataType& x) const
  {
    return map(std::multiplies<DataType>(), x);
  }

  /**
     Computes element-wise division with a scalar value.

     @param x A scalar divider value.
     @return The result vector.
   */
  Vector operator/(const DataType& x) const
  {
    return map(std::divides<DataType>(), x);
  }

  /**
     Computes element-wise summation with a scalar value.

     @param x A scalar value to add.
     @return The result vector.
   */
  Vector operator+(const DataType& x) const
  {
    return map(std::plus<DataType>(), x);
  }

  /**
     Computes element-wise subtraction with a scalar value.

     @param x A scalar value to subtract.
     @return The result vector.
   */
  Vector operator-(const DataType& x) const
  {
    return map(std::minus<DataType>(), x);
  }

  /**
     @return An iterator to the first element.
   */
  iterator begin() { return m_data.begin(); }

  /**
     @return An iterator to the end of the vector.
   */
  iterator end() { return m_data.end(); }

  /**
     @return A const iterator to the first element.
   */
  const_iterator begin() const { return m_data.begin(); }

  /**
     @return A const iterator to the end of the vector.
   */
  const_iterator end() const { return m_data.end(); }

  /**
     @return A reverse iterator to the first element.
   */
  reverse_iterator rbegin() { return m_data.rbegin(); }

  /**
     @return A reverse iterator to the end of the vector.
   */
  reverse_iterator rend() { return m_data.rend(); }

  /**
     @return A reverse const iterator to the first element.
   */
  const_reverse_iterator rbegin() const { return m_data.rbegin(); }

  /**
     @return A reverse const iterator to the end of the vector.
   */
  const_reverse_iterator rend() const { return m_data.rend(); }

  /**
     @return A reference to the first element.
   */
  reference front() { return m_data.front(); }

  /**
     @return A const reference to the first element.
  */
  const_reference front() const { return m_data.front(); }

  /**
     @return A reference to the last element.
  */
  reference back() { return m_data.back(); }

  /**
     @return A const reference to the last element.
  */
  const_reference back() const { return m_data.back(); }

  /**
     @return A string representation of the vector contents.
   */
  std::string tostring() const { return util::join_array(m_data, ", "); }
};

template <typename DataType>
inline std::ostream& operator<<(std::ostream& os, const Vector<DataType>& v)
{
  std::stringstream ss;
  ss << "{" << v.tostring() << "}";
  return os << ss.str();
}

using IntVector = Vector<int>;
using IndexVector = Vector<index_t>;

}  // namespace distconv
