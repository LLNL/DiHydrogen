#pragma once

#include <string>
#include <vector>
#include <array>

#include "distconv_config.hpp"
#include "distconv/util/util.hpp"
#include "distconv/util/util_mpi.hpp"

#ifdef __CUDACC__
#define HOST_DEV_FUNC __host__ __device__
#else
#define HOST_DEV_FUNC
#endif

namespace distconv {

using index_t = size_t;
using index_vector = std::vector<index_t>;
using int_vector = std::vector<int>;

enum Side {LHS=0, RHS=1};
const Side SIDES[2] = {LHS, RHS};

inline Side operator~(const Side &s) {
  return static_cast<Side>((~static_cast<int>(s)) & 1);
}

inline std::ostream& operator<<(std::ostream &os, const Side &s) {
  if (s == LHS) {
    return os << "LHS";
  } else {
    return os << "RHS";
  }
}

template <int ND, typename F>
void apply_to_sides(F &&f) {
  for (int i = 0; i < ND; ++i) {
    for (Side side: SIDES) {
      f(i, side);
    }
  }
}

template <typename F>
void apply_to_sides(int num_dims, F &&f) {
  for (int i = 0; i < num_dims; ++i) {
    for (Side side: SIDES) {
      f(i, side);
    }
  }
}

template <typename T>
struct BoundaryAttributes {
  std::array<T, 2> m_attrs;
  BoundaryAttributes() = default;
  BoundaryAttributes(const T &v): BoundaryAttributes(v, v) {}
  BoundaryAttributes(const T &lhs, const T &rhs):
      m_attrs({0 == LHS ? lhs : rhs, 1 == RHS ? rhs : lhs}) {}

  T &operator()(Side s) {
    return m_attrs[s];
  }
  const T &operator()(Side s) const {
    return m_attrs[s];
  }
  T *data() {
    return m_attrs.data();
  }
  const T *data() const{
    return m_attrs.data();
  }
};

template <typename F>
void apply_to_spatial_sides(int num_dims, F &&f) {
  for (int i = 0; i < num_dims - 2; ++i) {
    for (Side side: SIDES) {
      f(i, side);
    }
  }
}

template <int ND, typename F>
void apply_to_spatial_sides(F &&f) {
  for (int i = 0; i < ND - 2; ++i) {
    for (Side side: SIDES) {
      f(i, side);
    }
  }
}

template <int ND, typename T>
struct SpatialAttributes {
  constexpr static int NSD = ND - 2;
  T m_attrs[NSD][2];
  SpatialAttributes() {}
  SpatialAttributes(const T &attr_lhs, const T &attr_rhs) {
    apply_to_spatial_sides<ND>([&](int i, Side side) {
        (*this)(i, side) = side == LHS ? attr_lhs : attr_rhs;
      });
  }
  SpatialAttributes &operator=(const SpatialAttributes<ND, T> &sa) {
    apply_to_spatial_sides<ND>([&](int i, Side side) {
        (*this)(i, side) = sa(i, side);
      });
  }
  T &operator()(int d, Side s) {
    assert_always(d < NSD);
    return m_attrs[d][s];
  }
  const T &operator()(int d, Side s) const {
    assert_always(d < NSD);
    return m_attrs[d][s];
  }
  T *operator()(int d) {
    assert_always(d < NSD);
    return m_attrs[d];
  }
};

template <typename T>
class BoundaryAttributesV {
 private:
  BoundaryAttributes<T> m_default;
  std::vector<BoundaryAttributes<T>> m_attrs;

 public:
  BoundaryAttributesV() = default;

  BoundaryAttributesV(const T &lhs, const T &rhs):
      m_default(lhs, rhs) {}

  BoundaryAttributesV(const T &attr):
      BoundaryAttributesV(attr, attr) {}

  BoundaryAttributesV &operator=(const BoundaryAttributesV &sa) {
    m_default = sa.m_default;
    m_attrs = sa.m_attrs;
    return *this;
  }

  T &operator()(int d, Side s) {
    expand_if_needed(d);
    return m_attrs.at(d)(s);
  }

  const T &operator()(int d, Side s) const {
    return m_attrs.at(d)(s);
  }

  T *operator()(int d) {
    expand_if_needed(d);
    return m_attrs.at(d).data();
  }

  const T *operator()(int d) const {
    return m_attrs.at(d).data();
  }

  void clear() {
    m_attrs.clear();
  }

 private:
  void expand_if_needed(int d) {
    if (d >= (int)m_attrs.size()) {
      m_attrs.resize(d+1, m_default);
    }
  }
};

enum class HaloExchangeMethod {
  MPI, AL,
#ifdef DISTCONV_HAS_P2P
  P2P, HYBRID,
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  NVSHMEM, NVSHMEM_GRAPH, NVSHMEM_DIRECT, NVSHMEM_FUSED_NOTIFY
#endif // DISTCONV_HAS_NVSHMEM
};

inline std::ostream& operator<<(std::ostream &os, const HaloExchangeMethod &m) {
  if (m == HaloExchangeMethod::MPI) {
    return os << "MPI";
  } else if (m == HaloExchangeMethod::AL) {
    return os << "AL";
#ifdef DISTCONV_HAS_P2P
  } else if (m == HaloExchangeMethod::P2P) {
    return os << "P2P";
  } else if (m == HaloExchangeMethod::HYBRID) {
    return os << "HYBRID";
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  } else if (m == HaloExchangeMethod::NVSHMEM) {
    return os << "NVSHMEM";
  } else if (m == HaloExchangeMethod::NVSHMEM_GRAPH) {
    return os << "NVSHMEM_GRAPH";
  } else if (m == HaloExchangeMethod::NVSHMEM_DIRECT) {
    return os << "NVSHMEM_DIRECT";
  } else if (m == HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY) {
    return os << "NVSHMEM_FUSED_NOTIFY";
#endif // DISTCONV_HAS_NVSHMEM
  } else {
    util::PrintStreamError() << "Unknown halo exchange method";
    std::abort();
  }
}

inline HaloExchangeMethod GetHaloExchangeMethod(const std::string &method) {
  if (method == "MPI") {
    return HaloExchangeMethod::MPI;
  } else if (method == "AL") {
    return HaloExchangeMethod::AL;
#ifdef DISTCONV_HAS_P2P
  } else if (method == "P2P") {
    return HaloExchangeMethod::P2P;
  } else if (method == "HYBRID") {
    return HaloExchangeMethod::HYBRID;
#endif // DISTCONV_HAS_P2P
#ifdef DISTCONV_HAS_NVSHMEM
  } else if (method == "NVSHMEM") {
    return HaloExchangeMethod::NVSHMEM;
  } else if (method == "NVSHMEM_GRAPH") {
    return HaloExchangeMethod::NVSHMEM_GRAPH;
  } else if (method == "NVSHMEM_DIRECT") {
    return HaloExchangeMethod::NVSHMEM_DIRECT;
  } else if (method == "NVSHMEM_FUSED_NOTIFY") {
    return HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY;
#endif // DISTCONV_HAS_NVSHMEM
  } else {
    util::PrintStreamError() << "Unknown method name for halo exchange: " << method;
    std::abort();
  }
}

inline bool IsNVSHMEMUsed(HaloExchangeMethod m) {
#ifdef DISTCONV_HAS_NVSHMEM
  switch (m) {
    case HaloExchangeMethod::NVSHMEM:
    case HaloExchangeMethod::NVSHMEM_GRAPH:
    case HaloExchangeMethod::NVSHMEM_DIRECT:
    case HaloExchangeMethod::NVSHMEM_FUSED_NOTIFY:
      return true;
    default:
      return false;
  }
#else
  return false;
#endif
}

enum class ShuffleMethod {
  MPI, AL,
#ifdef DISTCONV_HAS_P2P
  P2P, HYBRID
#endif // DISTCONV_HAS_P2P
};

inline std::ostream& operator<<(std::ostream &os, const ShuffleMethod &m) {
  if (m == ShuffleMethod::MPI) {
    return os << "MPI";
  } else if (m == ShuffleMethod::AL) {
    return os << "AL";
#ifdef DISTCONV_HAS_P2P
  } else if (m == ShuffleMethod::P2P) {
    return os << "P2P";
  } else if (m == ShuffleMethod::HYBRID) {
    return os << "HYBRID";
#endif // DISTCONV_HAS_P2P
  } else {
    util::PrintStreamError() << "Unknown shuffle method";
    std::abort();
  }
}

enum class ChannelParallelismAlgorithm {NONE, AUTO, X, Y, W};

inline std::ostream& operator<<(std::ostream& os, const ChannelParallelismAlgorithm &a) {
  if (a == ChannelParallelismAlgorithm::NONE) {
    return os << "NONE";
  } else if (a == ChannelParallelismAlgorithm::AUTO) {
    return os << "AUTO";
  } else if (a == ChannelParallelismAlgorithm::X) {
    return os << "X";
  } else if (a == ChannelParallelismAlgorithm::Y) {
    return os << "Y";
  } else if (a == ChannelParallelismAlgorithm::W) {
    return os << "W";
  } else {
    util::PrintStreamError() << "Unknown channel parallelism algorithm";
    std::abort();
  }
}

enum class BatchnormImpl {
  MPI, AL_NCCL,
#ifdef DISTCONV_HAS_NVSHMEM
  NVSHMEM_NATIVE,
  NVSHMEM_RECURSIVE_DOUBLING_HOST,
  NVSHMEM_RECURSIVE_DOUBLING,
  NVSHMEM_RECURSIVE_DOUBLING_BUFFERED,
  NVSHMEM_RECURSIVE_DOUBLING_BLOCK,
  FUSED_NVSHMEM_RECURSIVE_DOUBLING,
#endif // DISTCONV_HAS_NVSHMEM
};

inline std::ostream& operator<<(std::ostream &os, const BatchnormImpl &v) {
  if (v == BatchnormImpl::MPI) {
    return os << "MPI";
  } else if (v == BatchnormImpl::AL_NCCL) {
    return os << "AL_NCCL";
#ifdef DISTCONV_HAS_NVSHMEM
  } else if (v == BatchnormImpl::NVSHMEM_NATIVE) {
    return os << "NVSHMEM_RECURSIVE_NATIVE";
  } else if (v == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_HOST) {
    return os << "NVSHMEM_RECURSIVE_DOUBLING_HOST";
  } else if (v == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING) {
    return os << "NVSHMEM_RECURSIVE_DOUBLING";
  } else if (v == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BUFFERED) {
    return os << "NVSHMEM_RECURSIVE_DOUBLING_BUFFERED";
  } else if (v == BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BLOCK) {
    return os << "NVSHMEM_RECURSIVE_DOUBLING_BLOCK";
  } else if (v == BatchnormImpl::FUSED_NVSHMEM_RECURSIVE_DOUBLING) {
    return os << "FUSED_NVSHMEM_RECURSIVE_DOUBLING";
#endif // DISTCONV_HAS_NVSHMEM
  } else {
    util::PrintStreamError() << "Unknown batchnorm implementation";
    std::abort();
  }
}

inline BatchnormImpl GetBatchnormImpl(const std::string &impl) {
  if (impl == "MPI") {
    return BatchnormImpl::MPI;
  } else if (impl == "AL_NCCL") {
    return BatchnormImpl::AL_NCCL;
#ifdef DISTCONV_HAS_NVSHMEM
  } else if (impl == "NVSHMEM_NATIVE") {
    return BatchnormImpl::NVSHMEM_NATIVE;
  } else if (impl == "NVSHMEM_RECURSIVE_DOUBLING_HOST") {
    return BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_HOST;
  } else if (impl == "NVSHMEM_RECURSIVE_DOUBLING") {
    return BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING;
  } else if (impl == "NVSHMEM_RECURSIVE_DOUBLING_BUFFERED") {
    return BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BUFFERED;
  } else if (impl == "NVSHMEM_RECURSIVE_DOUBLING_BLOCK") {
    return BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BLOCK;
  } else if (impl == "FUSED_NVSHMEM_RECURSIVE_DOUBLING") {
    return BatchnormImpl::FUSED_NVSHMEM_RECURSIVE_DOUBLING;
#endif // DISTCONV_HAS_NVSHMEM
  } else {
    util::PrintStreamError() << "Unknown implementation name for batchnorm: " << impl;
    std::abort();
  }
}

inline bool IsNVSHMEMUsed(BatchnormImpl m) {
#ifdef DISTCONV_HAS_NVSHMEM
  switch (m) {
    case BatchnormImpl::NVSHMEM_NATIVE:
    case BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_HOST:
    case BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING:
    case BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BUFFERED:
    case BatchnormImpl::NVSHMEM_RECURSIVE_DOUBLING_BLOCK:
    case BatchnormImpl::FUSED_NVSHMEM_RECURSIVE_DOUBLING:
      return true;
    default:
      return false;
  }
#else
  return false;
#endif
}

} //namespace distconv
