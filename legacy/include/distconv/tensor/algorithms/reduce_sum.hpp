#pragma once

#include "distconv/tensor/algorithms/common.hpp"
#include "distconv/tensor/tensor.hpp"

#include <type_traits>

namespace distconv
{
namespace tensor
{
namespace algorithms
{}  // namespace algorithms

template <typename DataType, typename Locale, typename Allocator>
typename std::enable_if<std::is_same<Allocator, BaseAllocator>::value,
                        int>::type
ReduceSum(const Tensor<DataType, Locale, Allocator>& src,
          Tensor<DataType, Locale, Allocator>& dst)
{
  // TODO
  return 0;
}

}  // namespace tensor
}  // namespace distconv
