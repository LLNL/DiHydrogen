#pragma once

#include <distconv_config.hpp>

#include "distconv/tensor/memory_gpu.hpp"
#include "distconv/tensor/tensor.hpp"
#include "distconv/tensor/tensor_process.hpp"
#include "distconv/util/util.hpp"

namespace distconv
{
namespace tensor
{

class LocaleCUDA
{
public:
  LocaleCUDA(int device, int num_devices, std::initializer_list<int> l)
    : m_device(device), m_num_devices(num_devices)
  {
    assert_always(num_devices == (int) l.size());
    int i = 0;
    for (auto it = l.begin(); it != l.end(); ++it, ++i)
    {
      if (device == *it)
      {
        m_rank = i;
      }
    }
  }

  int get_rank() const { return m_rank; }

  int get_size() const { return m_num_devices; }

  IndexVector get_rank_idx(const Distribution& dist) const
  {
    IndexVector rank_idx(dist.num_dims());
    const auto& locale_shape = dist.get_locale_shape();
    int rank = m_rank;
    for (int i = 0; i < dist.num_dims(); ++i)
    {
      rank_idx[i] = rank % locale_shape[i];
      rank = rank / locale_shape[i];
    }
    return rank_idx;
  }

protected:
  int m_device;
  int m_num_devices;
  int m_rank;
};

template <typename DataType, typename Allocator>
class TensorImpl<Tensor<DataType, LocaleCUDA, Allocator>>
{
  using TensorType = Tensor<DataType, LocaleCUDA, Allocator>;

public:
  TensorImpl() : m_tensor(nullptr) {}

  TensorImpl(TensorType* tensor) : m_tensor(tensor)
  {
    if (m_tensor)
    {
      init_proc_idx();
      init_local_tensor();
    }
  }

  TensorImpl(const TensorImpl<TensorType>& x) : TensorImpl(x.m_tensor) {}

  TensorImpl(TensorType* tensor, const TensorImpl<TensorType>& x)
    : TensorImpl(tensor)
  {}

  TensorImpl<TensorType>& operator=(const TensorImpl<TensorType>& x)
  {
    m_tensor = x.m_tensor;
    m_proc_idx = IndexVector();
    m_local_shape = Shape();
    m_local_real_shape = Shape();
    m_offset = 0;
    if (m_tensor)
    {
      init_proc_idx();
      init_local_tensor();
    }
  }

  Shape get_local_shape(bool include_halo) const
  {
    return include_halo ? m_local_real_shape : m_local_shape;
  }

  Shape get_local_shape() const { return get_local_shape(false); }

  size_t get_local_size() const { return get_local_shape().size(); }

  size_t get_local_real_size() const
  {
    return get_local_shape(true).get_size();
  }

  int allocate()
  {
    const auto& dist = m_tensor->get_distribution();
    const auto& locale_shape = dist.get_locale_shape();
    // MPI num procs must be equal to the locale shape size
    util::PrintStreamDebug()
      << "locale size: " << m_tensor->get_locale().get_size()
      << ", locale: " << locale_shape
      << ", shape size: " << locale_shape.size();
    assert_always((index_t) m_tensor->get_locale().get_size()
                  == locale_shape.size());
    size_t num_local_elements = get_local_real_size();
    util::PrintStreamDebug() << m_tensor->m_locale.get_rank() << ": "
                             << "num_local_elements: " << num_local_elements;
    assert_always(num_local_elements > 0);
    m_tensor->m_data.allocate(num_local_elements * sizeof(DataType),
                              get_local_real_shape()[0] * sizeof(DataType));
    return 0;
  }

  void nullify() { m_tensor->m_data.nullify(); }

  Shape get_local_real_shape() const { return get_local_shape(true); }

  index_t get_global_index(int dim, index_t local_idx) const
  {
    return m_offset[dim] + local_idx;
  }

  index_t get_local_offset(const IndexVector& idx, bool idx_include_halo) const
  {
    auto real_idx = idx;
    if (!idx_include_halo)
    {
      real_idx = real_idx + m_tensor->get_distribution().get_overlap();
    }
    return get_offset(real_idx, get_local_real_shape(), m_tensor->get_pitch());
  }

protected:
  // REFACTORING: duplicate with MPI TensorImpl
  void init_proc_idx()
  {
    const int nd = m_tensor->get_num_dims();
    const auto& dist = m_tensor->get_distribution();
    const auto& locale_shape = dist.get_locale_shape();
    index_t rank = m_tensor->get_locale().get_rank();
    m_proc_idx = IndexVector(nd);
    for (int i = 0; i < nd; ++i)
    {
      m_proc_idx[i] = rank % locale_shape[i];
      rank = rank / locale_shape[i];
    }
    util::PrintStreamDebug() << m_tensor->get_locale().get_rank() << ": "
                             << "proc_idx: " << m_proc_idx;
  }

  // REFACTORING: duplicate with MPI TensorImpl
  void init_local_tensor()
  {
    const int nd = m_tensor->get_num_dims();
    const auto& tensor_shape = m_tensor->get_shape();
    const auto& dist = m_tensor->get_distribution();
    const auto& locale_shape = dist.get_locale_shape();
    m_local_shape = Shape(nd);
    m_local_real_shape = Shape(nd);
    m_offset = IndexVector(nd);
    for (int i = 0; i < nd; ++i)
    {
      size_t proc_chunk_size;
      size_t real_size_extra = 0;
      if (dist.is_distributed(i))
      {
        proc_chunk_size = tensor_shape[i] / locale_shape[i];
        // TODO: non-divisible case
        assert_always((tensor_shape[i] % locale_shape[i]) == 0);
        // Add halo regions
        real_size_extra = dist.get_overlap(i) * 2;
      }
      else
      {
        proc_chunk_size = tensor_shape[i];
      }
      m_local_shape[i] = proc_chunk_size;
      m_local_real_shape[i] = proc_chunk_size + real_size_extra;
      m_offset[i] = proc_chunk_size * m_proc_idx[i];
    }
    util::PrintStreamDebug() << "local shape: " << m_local_shape
                             << ", local real shape: " << m_local_real_shape;
  }

  TensorType* m_tensor;
  IndexVector m_proc_idx;
  Shape m_local_shape;
  Shape m_local_real_shape;
  IndexVector m_offset;
};

} // namespace tensor
} // namespace distconv
