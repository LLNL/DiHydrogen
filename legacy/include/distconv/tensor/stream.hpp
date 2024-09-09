#pragma once

#include "distconv/base.hpp"

namespace distconv
{
namespace tensor
{

template <typename Allocator>
struct Stream;

struct DefaultStream
{
  DefaultStream() = default;
  DefaultStream(int v) {}
  static DefaultStream value;
};

} // namespace tensor
} // namespace distconv
