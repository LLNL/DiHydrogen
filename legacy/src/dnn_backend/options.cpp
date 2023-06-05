////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include "distconv/dnn_backend/dnn_backend.hpp"
#include "distconv/util/util_mpi.hpp" // MPIRootPrintStreamDebug

namespace distconv
{

Options::Options(bool overlap_halo_exchange_in,
                 bool deterministic_in,
                 bool enable_profiling_in,
                 float ws_capacity_factor_in)
    : overlap_halo_exchange{overlap_halo_exchange_in},
      m_deterministic{deterministic_in},
      enable_profiling{enable_profiling_in},
      ws_capacity_factor{ws_capacity_factor_in}
{
    // FIXME (trb): This carries over the previous logic, which is
    // BAD. `DISTCONV_OVERLAP_HALO_EXCHANGE=0` is still "detected", so
    // `overlap_halo_exchange` will be set to `true`. But, I don't
    // want to change any existing behavior.
    if (std::getenv("DISTCONV_OVERLAP_HALO_EXCHANGE"))
    {
        util::MPIRootPrintStreamDebug() << "Environment variable: "
                                        << "DISTCONV_OVERLAP_HALO_EXCHANGE"
                                        << " detected";
        overlap_halo_exchange = true;
    }
    if (std::getenv("DISTCONV_DETERMINISTIC"))
    {
        util::MPIRootPrintStreamDebug() << "Environment variable: "
                                        << "DISTCONV_DETERMINISTIC"
                                        << " detected";
        m_deterministic = true;
    }
    if (std::getenv("DISTCONV_ENABLE_PROFILING"))
    {
        util::MPIRootPrintStreamDebug() << "Environment variable: "
                                        << "DISTCONV_ENABLE_PROFILING"
                                        << " detected";
        enable_profiling = true;
    }
    if (std::getenv("DISTCONV_WS_CAPACITY_FACTOR"))
    {
        util::MPIRootPrintStreamDebug() << "Environment variable: "
                                        << "DISTCONV_WS_CAPACITY_FACTOR"
                                        << " detected";
        ws_capacity_factor = atof(std::getenv("DISTCONV_WS_CAPACITY_FACTOR"));
    }
}

} // namespace distconv
