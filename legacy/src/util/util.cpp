#include "distconv/util/util.hpp"
#include <fstream>
#include <sstream>

namespace distconv {
namespace util {

template <>
const std::string to_string<std::string>(const std::string i) {
  return i;
}

int get_memory_usage() {
  std::ifstream proc_status("/proc/self/status");
  std::string line;
  int vmsize;
  while (std::getline(proc_status, line)) {
    if (line.find("VmSize") != 0) continue;
    std::istringstream iss(line);
    std::string key;
    iss >> key >> vmsize;
    break;
  }
  return vmsize / (1000 * 1000);
}

} // namespace util
} // namespace distconv
