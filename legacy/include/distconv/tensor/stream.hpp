#pragma once

namespace distconv {
namespace tensor {

template <typename Allocator>
struct Stream;

struct DefaultStream {
  DefaultStream() = default;
  DefaultStream(int v) {}
  static DefaultStream value;
};

} // namespace tensro
} // namespae distconv

