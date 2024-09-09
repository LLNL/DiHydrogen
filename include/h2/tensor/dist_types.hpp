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

#include <El.hpp>

#include "tensor_types.hpp"

namespace h2
{

#ifndef HYDROGEN_HAVE_ALUMINUM
#error "DiHydrogen distributed tensors require Aluminum support in Hydrogen"
#endif
/**
 * Wrapper around communicators for various Aluminum backends.
 */
using Comm = El::mpi::Comm; // Use Hydrogen's communicator wrappers for now.

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
inline std::ostream& operator<<(std::ostream& os, const Distribution& dist)
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

using DistTTuple = DistributionTypeTuple; // Alias to save some typing.

/** Type used for representing ranks in communicators/grids. */
using RankType = std::int32_t;

} // namespace h2
