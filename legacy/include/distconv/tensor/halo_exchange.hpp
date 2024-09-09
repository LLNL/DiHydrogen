#pragma once

#include "distconv/tensor/stream.hpp"

namespace distconv
{
namespace tensor
{

enum class HaloExchangeAccumOp
{
  ID,
  SUM,
  MAX,
  MIN
};

template <typename DataType, typename Allocator, typename AlBackend>
class HaloExchange;

}  // namespace tensor
}  // namespace distconv
