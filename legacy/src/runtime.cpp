#include "distconv/runtime.hpp"

#include <cstdlib>

namespace distconv {

namespace {
bool initialized = false;
Config cfg;
}

void initialize() {
  if (!initialized) {
    cfg.m_nvtx = std::getenv("DISTCONV_NVTX") != nullptr;
    initialized = true;
  }
}

const Config &get_config() {
  initialize();
  return cfg;
}

} // namespace
