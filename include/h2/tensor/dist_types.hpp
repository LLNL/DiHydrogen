////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Types for distributed tensors.
 */

#include <Al.hpp>

#include "tensor_types.hpp"

namespace h2
{

/** @class Comm
 *  @brief Minimal interface around Aluminum's communicator
 *  @todo Flesh this out once the communication requirements and
 *        semantics of H2 become clearer.
 */
class Comm
{
  std::unique_ptr<typename Al::MPIBackend::comm_type> m_al_comm;

public:
  /** Construct a null communicator */
  Comm() : Comm{MPI_COMM_NULL} {}

  /** Construct from an MPI communicator */
  Comm(MPI_Comm comm)
  {
    if (comm != MPI_COMM_NULL)
    {
      m_al_comm = std::make_unique<typename Al::MPIBackend::comm_type>(comm);
    }
  }

  Comm(Comm&&) = default;
  Comm& operator=(Comm&&) = default;

  // Copy semantics are hard.
  Comm(Comm const&) = delete;
  Comm& operator=(Comm const&) = delete;

  /** Get the rank in the comm */
  int rank() const
  {
    if (!m_al_comm)
      throw H2Exception("Comm is NULL");
    return m_al_comm->rank();
  }

  /** Get the size of the comm */
  int size() const
  {
    if (!m_al_comm)
      throw H2Exception("Comm is NULL");
    return m_al_comm->size();
  }

  /** Get the underlying MPI_Comm handle */
  MPI_Comm get_mpi_handle() const noexcept
  {
    if (m_al_comm)
      return m_al_comm->get_comm();
    else
      return MPI_COMM_NULL;
  }

  /**
   * Get the underlying Aluminum communicator.
   *
   * Note that Aluminum takes communicators by non-const reference,
   * hence the (very intentional) return type here. However, 'this'
   * does not logically mutate because of this operation, so the
   * function itself is marked 'const'.
   **/
  typename Al::MPIBackend::comm_type& get_al_comm() const
  {
    if (!m_al_comm)
      throw H2Exception("Comm is NULL");
    return *m_al_comm;
  }
};

/** @brief Get the (global) communicator for MPI_COMM_WORLD */
inline Comm const& get_comm_world()
{
  static Comm comm{MPI_COMM_WORLD};
  return comm;
}

/**
 * Defines how a dimension of a tensor is distributed on a processor
 * grid.
 */
enum class Distribution
{
  Undefined,  /**< No defined distribution. */
  Block,      /**< A block distribution with same-sized blocks. */
  Replicated, /**< Data is replicated. */
  Single      /**< Data resides on a single processor. */
};

/** Support printing Distribution. */
inline std::ostream& operator<<(std::ostream& os, Distribution const& dist)
{
  switch (dist)
  {
  case Distribution::Undefined: os << "Undefined"; break;
  case Distribution::Block: os << "Block"; break;
  case Distribution::Replicated: os << "Replicated"; break;
  case Distribution::Single: os << "Single"; break;
  default: os << "Unknown"; break;
  }
  return os;
}

/** Tuple of distributions. */
using DistributionTypeTuple = NDimTuple<Distribution>;

using DistTTuple = DistributionTypeTuple;  // Alias to save some typing.

/** Type used for representing ranks in communicators/grids. */
using RankType = std::int32_t;

}  // namespace h2
