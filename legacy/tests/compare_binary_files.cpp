#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <cstdlib>

#include "distconv/util/util.hpp"

template <typename T>
size_t compare(T threshold, std::string path1, std::string path2) {
  std::ifstream f1, f2;
  f1.open(path1.c_str(), std::ios::binary);
  if (!f1.is_open()) {
    std::cerr << "Opening " << path1 << " failed.\n";
    std::exit(1);
  }
  f2.open(path2.c_str(), std::ios::binary);
  if (!f2.is_open()) {
    std::cerr << "Opening " << path2 << " failed.\n";
    std::exit(1);
  }

  f1.seekg(0, f1.end);
  auto len1 = f1.tellg();
  f2.seekg(0, f2.end);
  auto len2 = f2.tellg();
  if (len1 != len2) {
    std::cerr << "File length not equal.\n";
    std::exit(1);
  }

  f1.seekg(0);
  f2.seekg(0);
  T max_diff = 0;
  size_t mismatch_count = 0;
  while (true) {
    T v1, v2;
    f1.read((char*)&v1, sizeof(T));
    f2.read((char*)&v2, sizeof(T));
    if (f1.eof()) {
      break;
    }
    if (v1 == v2) continue;
    auto d = std::abs(v1 - v2);
    auto r = std::max(std::abs(d / v1), std::abs(d / v2));
    if (r <= threshold) continue;
    ++mismatch_count;
    max_diff = std::max(max_diff, r);
  }
  f1.close();
  f2.close();
  std::cout << "#errors: " << mismatch_count << ", maximum error ratio: "
            << max_diff << std::endl;
  return mismatch_count;
}

int main(int argc, char *argv[]) {
  assert_eq(argc, 5);
  std::string type_str(argv[1]);
  std::string threshold(argv[2]);
  std::string path1(argv[3]);
  std::string path2(argv[4]);
  size_t ret = 0;
  if (type_str == "float") {
    ret = compare<float>(std::stof(threshold), path1, path2);
  } else if (type_str == "double") {
    ret = compare<double>(std::stof(threshold), path1, path2);
  } else {
    std::cerr << "Unknown type: " << type_str << "\n";
    ret = -1;
  }
  if (ret) {
    std::cerr << "Error detected\n";
    std::exit(1);
  }
  return 0;
}
