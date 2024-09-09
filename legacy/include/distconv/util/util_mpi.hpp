#pragma once

#include "distconv/util/util.hpp"
#include "mpi.h"

#include <iostream>
#include <ostream>

#define DISTCONV_CHECK_MPI(call)                                               \
  do                                                                           \
  {                                                                            \
    int status = call;                                                         \
    if (status != MPI_SUCCESS)                                                 \
    {                                                                          \
      std::cerr << "MPI error" << std::endl;                                   \
      std::cerr << "Error at " << __FILE__ << ":" << __LINE__ << std::endl;    \
      MPI_Abort(MPI_COMM_WORLD, status);                                       \
    }                                                                          \
  } while (0)

namespace distconv
{
namespace util
{

template <typename T>
MPI_Datatype get_mpi_data_type();

template <>
inline MPI_Datatype get_mpi_data_type<char>()
{
  return MPI_CHAR;
}

template <>
inline MPI_Datatype get_mpi_data_type<unsigned char>()
{
  return MPI_UNSIGNED_CHAR;
}

template <>
inline MPI_Datatype get_mpi_data_type<short>()
{
  return MPI_SHORT;
}

template <>
inline MPI_Datatype get_mpi_data_type<unsigned short>()
{
  return MPI_UNSIGNED_SHORT;
}

template <>
inline MPI_Datatype get_mpi_data_type<int>()
{
  return MPI_INT;
}

template <>
inline MPI_Datatype get_mpi_data_type<unsigned int>()
{
  return MPI_UNSIGNED;
}

template <>
inline MPI_Datatype get_mpi_data_type<long>()
{
  return MPI_LONG;
}

template <>
inline MPI_Datatype get_mpi_data_type<unsigned long>()
{
  return MPI_UNSIGNED_LONG;
}

template <>
inline MPI_Datatype get_mpi_data_type<float>()
{
  return MPI_FLOAT;
}

template <>
inline MPI_Datatype get_mpi_data_type<double>()
{
  return MPI_DOUBLE;
}

inline MPI_Comm get_mpi_local_comm(MPI_Comm comm)
{
  MPI_Comm local_comm;
  MPI_Comm_split_type(
    comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
  return local_comm;
}

inline int get_mpi_comm_local_size(MPI_Comm comm)
{
  MPI_Comm local_comm = get_mpi_local_comm(comm);
  int local_comm_size;
  MPI_Comm_size(local_comm, &local_comm_size);
  MPI_Comm_free(&local_comm);
  return local_comm_size;
}

#ifdef DISTCONV_DEBUG
class MPIPrintStreamDebug : public PrintStreamDebug
{
public:
  MPIPrintStreamDebug() : PrintStreamDebug()
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }

protected:
  int m_rank;
};
#else
using MPIPrintStreamDebug = PrintStream<false>;
#endif

class MPIPrintStreamError : public PrintStreamError
{
public:
  MPIPrintStreamError() : PrintStreamError()
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }

protected:
  int m_rank;
};

class MPIPrintStreamInfo : public PrintStreamInfo
{
public:
  MPIPrintStreamInfo() : PrintStreamInfo()
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }

protected:
  int m_rank;
};

class MPIPrintStreamWarning : public PrintStreamWarning
{
public:
  MPIPrintStreamWarning() : PrintStreamWarning()
  {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
    m_prefix << "[" << m_rank << "] ";
  }

protected:
  int m_rank;
};

#ifdef DISTCONV_DEBUG
class MPIRootPrintStreamDebug : public MPIPrintStreamDebug
{
public:
  MPIRootPrintStreamDebug() : MPIPrintStreamDebug() { m_enable = m_rank == 0; }
};
#else
using MPIRootPrintStreamDebug = PrintStream<false>;
#endif

class MPIRootPrintStreamError : public MPIPrintStreamError
{
public:
  MPIRootPrintStreamError() : MPIPrintStreamError() { m_enable = m_rank == 0; }
};

class MPIRootPrintStreamInfo : public MPIPrintStreamInfo
{
public:
  MPIRootPrintStreamInfo() : MPIPrintStreamInfo() { m_enable = m_rank == 0; }
};

class MPIRootPrintStreamWarning : public MPIPrintStreamWarning
{
public:
  MPIRootPrintStreamWarning() : MPIPrintStreamWarning()
  {
    m_enable = m_rank == 0;
  }
};

}  // namespace util
}  // namespace distconv
