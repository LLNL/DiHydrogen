#pragma once

namespace distconv {

struct Config {
  bool m_nvtx;
};

void initialize();
const Config &get_config();

} // namespace distconv
