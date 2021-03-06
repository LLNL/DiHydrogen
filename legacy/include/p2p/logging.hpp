#pragma once

#include "distconv_config.hpp"

#include <iostream>
#include <sstream>

#include <mpi.h>

namespace p2p {
namespace logging {

template <bool ENABLE=true>
class PrintStream;

template <>
class PrintStream<false> {
 public:
  PrintStream(std::ostream &os=std::cerr) {}
  template <typename PrefixType>  
  PrintStream(std::ostream &os, const PrefixType &prefix) {}
  template <typename PrefixType>  
  PrintStream(const PrefixType &prefix) {}
  ~PrintStream() = default;
  template <typename X>
  PrintStream &operator<<(const X &) {
    return *this;
  }
};

template <>
class PrintStream<true> {
 public:
  PrintStream(std::ostream &os=std::cerr): m_os(os) {}
  template <typename PrefixType>  
  PrintStream(std::ostream &os, const PrefixType &prefix):
      m_os(os) {
    ss << prefix;
  }
  template <typename PrefixType>  
  PrintStream(const PrefixType &prefix): m_os(std::cerr) {
    ss << prefix;
  }
  ~PrintStream() {
    if (m_enable) {
      std::string msg = m_prefix.str() + ss.str();
      m_os << msg;
    }
  }
  std::stringstream &operator()() {
    return ss;
  }
  template <typename X>
  PrintStream<true> &operator<<(const X &x) {
    ss << x;
    return *this;
  }
  
 protected:
  bool m_enable = true;
  std::ostream &m_os;
  std::stringstream ss;
  std::stringstream m_prefix;
};

#ifdef P2P_DEBUG
class PrintStreamDebug: public PrintStream<true> {
 public:
  PrintStreamDebug(): PrintStream(std::cerr, "[DEBUG] ") {}
};
#else
using PrintStreamDebug = PrintStream<false>;
#endif

class PrintStreamError: public PrintStream<true> {
 public:
  PrintStreamError(): PrintStream(std::cerr, "[ERROR] ") {}
};

class PrintStreamInfo: public PrintStream<true> {
 public:
  PrintStreamInfo(): PrintStream(std::cerr, "[INFO] ") {}
};

#ifdef P2P_DEBUG
class MPIPrintStreamDebug: public PrintStreamDebug {
 public:
  MPIPrintStreamDebug(): PrintStreamDebug() {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }
 protected:
  int m_rank;
};
#else
using MPIPrintStreamDebug = PrintStream<false>;
#endif

class MPIPrintStreamError: public PrintStreamError {
 public:
  MPIPrintStreamError(): PrintStreamError() {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }
 protected:
  int m_rank;
};

class MPIPrintStreamInfo: public PrintStreamInfo {
 public:
  MPIPrintStreamInfo(): PrintStreamInfo() {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }
 protected:
  int m_rank;
};

#ifdef P2P_DEBUG
class MPIRootPrintStreamDebug: public MPIPrintStreamDebug {
 public:
  MPIRootPrintStreamDebug(): MPIPrintStreamDebug() {
    m_enable = m_rank == 0;
  }
};
#else
using MPIRootPrintStreamDebug = PrintStream<false>;
#endif

class MPIRootPrintStreamError: public MPIPrintStreamError {
 public:
  MPIRootPrintStreamError(): MPIPrintStreamError() {
    m_enable = m_rank == 0;    
  }
};

class MPIRootPrintStreamInfo: public MPIPrintStreamInfo {
 public:
  MPIRootPrintStreamInfo(): MPIPrintStreamInfo() {
    m_enable = m_rank == 0;        
  }
};

} // namespace logging
} // namespace p2p
