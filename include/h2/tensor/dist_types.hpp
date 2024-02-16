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


namespace h2
{

#ifndef HYDROGEN_HAVE_ALUMINUM
#error "DiHydrogen distributed tensors require Aluminum support in Hydrogen"
#endif
/**
 * Wrapper around communicators for various Aluminum backends.
 */
using Comm = El::mpi::Comm;  // Use Hydrogen's communicator wrappers for now.

}  // namespace h2
