#pragma once

#include "distconv/base.hpp"

namespace distconv
{
namespace tensor
{

template <typename DataType>
class Allreduce
{
public:
  Allreduce() = default;
  virtual ~Allreduce() = default;

  virtual void
  allreduce(const DataType* sendbuf, DataType* recvbuf, size_t count) = 0;
  virtual void allreduce(DataType* buf, size_t count)
  {
    allreduce(buf, buf, count);
  }
};

} // namespace tensor
} // namespace distconv
