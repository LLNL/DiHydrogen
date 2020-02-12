#pragma once

#include <iostream>
#include <ostream>
#include <sstream>
#include <cstring>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <memory>

#include "distconv_config.hpp"

// Preprocessors can be confused if an expression contains curly
// braces and considers an expression is separated at the braces. A
// workaaround is to use __VA_ARGS__. See https://stackoverflow.com/questions/20913103/is-it-possible-to-pass-a-brace-enclosed-initializer-as-a-macro-parameter
#define assert_always(...) do {                         \
    if ((__VA_ARGS__) == 0) {                           \
      std::stringstream ss;                             \
      ss << __FILE__ << ":" << __LINE__                 \
         << ": " << __func__ << " Assertion "           \
         << #__VA_ARGS__ << " failed.\n";               \
      std::cerr << ss.str();                            \
      abort();                                          \
    } } while (0)

#define assert0(...) do {                               \
    if ((__VA_ARGS__) != 0) {                           \
      auto x = (__VA_ARGS__);                           \
      std::stringstream ss;                             \
      ss << __FILE__ << ":" << __LINE__                 \
         << ": " << __func__ << " Assertion "           \
         << #__VA_ARGS__ << " ("                        \
         << x << ") == 0 failed.\n";                    \
      std::cerr << ss.str();                            \
      abort();                                          \
    } } while (0)

#define assert_eq(x, y) do {                                     \
    auto x_result = (x);                                         \
    auto y_result = (y);                                         \
    if (x_result != y_result) {                                  \
      std::stringstream ss;                                      \
      ss << __FILE__ << ":" << __LINE__                          \
         << ": " << __func__ << " Assertion "                    \
         << #x << " (" << x_result << ") == "                    \
         << #y << " (" << y_result << ") failed.\n";             \
      std::cerr << ss.str();                                     \
      abort();                                                   \
    } } while (0)

#define assert_ne(x, y) do {                                     \
    auto x_result = (x);                                         \
    auto y_result = (y);                                         \
    if (x_result == y_result) {                                  \
      std::stringstream ss;                                      \
      ss << __FILE__ << ":" << __LINE__                          \
         << ": " << __func__ << " Assertion "                    \
         << #x << " (" << x_result << ") != "                    \
         << #y << " (" << y_result << ") failed.\n";             \
      std::cerr << ss.str();                                     \
      abort();                                                   \
    } } while (0)

namespace distconv {
namespace util {

template <typename I>
std::string join(const std::string &delim, const I &begin, const I &end) {
  std::stringstream ss;
  int first = true;
  for (auto it = begin; it != end; ++it) {
    if (!first) {
      ss << delim;
    } else {
      first = false;
    }
    ss << *it;
  }
  return ss.str();
}

template <typename I>
std::ostream &print_vector(std::ostream &os, const I &begin, const I &end) {
  os << "{" << join(", ", begin, end) << "}";
  return os;
}

template <typename I>
std::string tostring(const I &begin, const I &end) {
  std::stringstream ss;
  print_vector(ss, begin, end);
  return ss.str();
}

template <bool ENABLE=true>
class PrintStream;

template <>
class PrintStream<false> {
 public:
  PrintStream() {}
  template <typename PrefixType>
  PrintStream(bool enable, std::ostream &os, const PrefixType &prefix) {}
  ~PrintStream() = default;
  template <typename X>
  PrintStream<false> &operator<<(const X &) {
    return *this;
  }
  PrintStream<false> &operator<<(std::ostream&(*endl)(std::ostream&)) {
    return *this;
  }
};

template <>
class PrintStream<true> {
 public:
  template <typename PrefixType>
  PrintStream(bool enable, std::ostream &os, const PrefixType &prefix):
      m_enable(enable), m_os(os) {
    ss << prefix;
  }
  ~PrintStream() {
    if (m_enable) {
      if (!m_printed_newline) ss << std::endl;
      std::string msg = m_prefix.str() + ss.str();
      m_os << msg;
    }
  }
  std::stringstream &operator()() {
    return ss;
  }
  PrintStream<true> &operator<<(const char *x) {
    ss << x;
    m_printed_newline = x[std::strlen(x)-1] == '\n';
    return *this;
  }
  template <typename X>
  PrintStream<true> &operator<<(const X &x) {
    ss << x;
    return *this;
  }
  PrintStream<true> &operator<<(std::ostream&(*endl)(std::ostream&)) {
    ss << endl;
    m_printed_newline = true;
    return *this;
  }

 protected:
  bool m_enable;
  std::ostream &m_os;
  std::stringstream ss;
  std::stringstream m_prefix;
  bool m_printed_newline = false;
};

#ifdef DISTCONV_DEBUG
class PrintStreamDebug: public PrintStream<true> {
 public:
  PrintStreamDebug(const std::string &prefix="[DEBUG] "):
      PrintStream(true, std::cerr, prefix) {
    // Disable if DISTCONV_PRINT_DEBUG=0
    char *env = std::getenv("DISTCONV_PRINT_DEBUG");
    if (env) {
      int var = std::atoi(env);
      if (var == 0) {
        m_enable = false;
      }
    }
  }
};
#else
using PrintStreamDebug = PrintStream<false>;
#endif

class PrintStreamError: public PrintStream<true> {
 public:
  PrintStreamError(): PrintStream(true, std::cerr, "[ERROR] ") {
    // Disable if DISTCONV_PRINT_DEBUG=0
    char *env = std::getenv("DISTCONV_PRINT_ERROR");
    if (env) {
      int var = std::atoi(env);
      if (var == 0) {
        m_enable = false;
      }
    }
  }
};

class PrintStreamInfo: public PrintStream<true> {
 public:
  PrintStreamInfo(): PrintStream(true, std::cerr, "[INFO] ") {
    // Disable if DISTCONV_PRINT_DEBUG=0
    char *env = std::getenv("DISTCONV_PRINT_INFO");
    if (env) {
      int var = std::atoi(env);
      if (var == 0) {
        m_enable = false;
      }
    }
  }
};

class PrintStreamWarning: public PrintStream<true> {
 public:
  PrintStreamWarning(): PrintStream(true, std::cerr, "[WARNING] ") {
    // Disable if DISTCONV_PRINT_WARNING=0
    char *env = std::getenv("DISTCONV_PRINT_WARNING");
    if (env) {
      int var = std::atoi(env);
      if (var == 0) {
        m_enable = false;
      }
    }
  }
};

// Copied from https://stackoverflow.com/a/236803
template<typename Out>
inline void split(const std::string &s, char delim, Out result) {
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    *(result++) = item;
  }
}
// Copied from https://stackoverflow.com/a/236803
inline std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  split(s, delim, std::back_inserter(elems));
  return elems;
}

template <typename T>
inline T ceil(T x, T y) {
  return (x + y - T(1)) / y;
}

// Return whether all of the elements of `ary` is the same.
template<typename T>
inline bool is_all_elements_equal(const std::vector<T> &ary) {
  return std::equal(ary.begin()+1, ary.end(), ary.begin());
}

// Split `str` with spaces and parse each of the separated string as int.
template<typename T>
std::vector<T> split_spaced_array(const std::string &str) {
  const auto split =
      [](std::string s, const std::string delimiter) {
        std::vector<std::string> sary;
        while(true) {
          const auto pos = s.find(delimiter);
          if(pos != std::string::npos) {
            sary.push_back(s.substr(0, pos));
            s = s.substr(pos+1);
          } else {
            sary.push_back(s);
            break;
          }
        }
        return sary;
      };

  const std::vector<std::string> sary = split(str, ",");
  std::vector<T> ary;
  // cf. https://gist.github.com/mark-d-holmberg/862733
  for(const auto s : sary) {
    std::istringstream ss(s);
    T val;
    ss >> val;
    ary.push_back(val);
  }
  return ary;
}

// A wrapper for std::to_string, but also accepting std::string.
template <typename T>
const std::string to_string(const T i) {
        return std::to_string(i);
}
template <>
const std::string to_string<std::string>(const std::string i);

// Join each element of `ary` with `delimiter` into a single string.
template<typename V>
std::string join_array(const V &ary,
                       const std::string &delimiter) {
  if (ary.begin() == ary.end()) {
    return std::string("");
  } else {
    return std::accumulate(ary.begin()+1, ary.end(), to_string(ary[0]),
                           [delimiter](const std::string s,
                                       const typename V::value_type i) {
                             return s + delimiter + to_string(i);
                           });
  }
}

// Wrappers of `join_array` with specific delimiters.
template<typename T>
inline std::string join_spaced_array(const std::vector<T> &ary) {
  return join_array(ary, " ");
}
template<typename T>
inline std::string join_xd_array(const std::vector<T> &ary) {
  return join_array(ary, "x");
}

// Return the reversed vector of `v`.
template<typename V>
inline V reverse(const V v) {
  return V(v.rbegin(), v.rend());
}

inline void *aligned_malloc(size_t s) {
  unsigned long align_size = sysconf(_SC_PAGESIZE);
  void *p = nullptr;
  if (posix_memalign(&p, align_size, s)) {
    util::PrintStreamError() << "posix_memalign failed.";
  }
  return p;
}

int get_memory_usage();

template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts&&... params) {
  return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

} // namespace util
} // namespace distconv
