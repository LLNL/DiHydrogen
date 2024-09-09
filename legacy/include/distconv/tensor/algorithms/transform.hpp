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

template <typename Tensor, typename UnaryFunction>
typename std::enable_if<
  std::is_same<typename Tensor::allocator_type, BaseAllocator>::value,
  int>::type
Transform(Tensor& tensor, UnaryFunction op)
{
  // TODO
  return 0;
}

}  // namespace tensor
}  // namespace distconv
