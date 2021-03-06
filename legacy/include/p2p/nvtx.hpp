#pragma once

#include "p2p/config.hpp"
#include "nvToolsExt.h"

namespace p2p {
namespace internal {

inline void nvtx_start(const char *id) {
  if (cfg.insert_nvtx_mark) {
    nvtxRangePushA(id);
  }
  return;
}

inline void nvtx_end() {
  if (cfg.insert_nvtx_mark) {
    nvtxRangePop();
  }
  return;
}

} // namespace internal
} // namespace p2p
