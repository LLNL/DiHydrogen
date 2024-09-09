#include "distconv/runtime.hpp"

#include <cstdlib>

namespace distconv
{

namespace
{
bool initialized = false;
Config cfg;
}  // namespace

void initialize()
{
  if (!initialized)
  {
    cfg.profiling = std::getenv("DISTCONV_PROFILING") != nullptr;
    if (!cfg.profiling)
      cfg.profiling = std::getenv("DISTCONV_NVTX") != nullptr;
    initialized = true;
  }
}

Config const& get_config()
{
  initialize();
  return cfg;
}

}  // namespace distconv
