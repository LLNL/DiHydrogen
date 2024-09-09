#pragma once

namespace distconv
{

struct Config
{
  bool profiling;
};

void initialize();
Config const& get_config();

}  // namespace distconv
