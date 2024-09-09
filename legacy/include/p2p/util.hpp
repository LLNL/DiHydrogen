#pragma once

#include "mpi.h"

#include <cstdlib>
#include <iostream>

#define P2P_ASSERT_ALWAYS(x)                                                   \
  do                                                                           \
  {                                                                            \
    if ((x) == 0)                                                              \
    {                                                                          \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " << __func__             \
                << " Assertion " << #x << " failed.\n";                        \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define P2P_ASSERT0(x)                                                         \
  do                                                                           \
  {                                                                            \
    if ((x) != 0)                                                              \
    {                                                                          \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " << __func__             \
                << " Assertion " << #x << " == 0 failed.\n";                   \
      std::abort();                                                            \
    }                                                                          \
  } while (0)

#define P2P_CHECK_MPI(call)                                                    \
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
