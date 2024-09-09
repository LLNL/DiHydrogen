#pragma once

namespace distconv {

struct Config {
    bool profiling;
};

void initialize();
const Config &get_config();

} // namespace distconv
