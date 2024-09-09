////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

/** @file
 *
 * Utilities to interface with environment variables.
 */

#include "h2/utils/strings.hpp"

#include <string>

namespace h2
{

/**
 * A note on environment variables:
 *
 * This provides an interface for getting environment variables and
 * coercing their value to a given type. This is a thin wrapper around
 * standard interfaces for accessing the environment.
 *
 * The main operations are in the `env` namespace, and are meant for
 * accessing H2-specific environment variables. The names of these
 * variables are always in uppercase and are prefixed with "H2_" (this
 * is done automatically). The value of these variables is cached after
 * the first access.
 *
 * These variables must be registered at compile-time in the
 * `environment_vars.cpp` file. This is to ensure all environment
 * variables are centralized and documented.
 *
 * This also provides a "raw" interface (in the `env::raw` namespace)
 * that is a direct wrapper around standard calls to access environment
 * variables. Names are not modified (i.e., "H2_" is not prepended) and
 * the values are not cached.
 */

namespace env
{

/**
 * Return true if the H2 environment variable name is set in the
 * environment.
 *
 * @note If the variable is not set, it will still have its default
 * value.
 */
bool exists(const std::string& name);

/**
 * Return the raw value (i.e., the exact string value of the variable)
 * of the H2 environment variable name.
 *
 * @note This may be the default value.
 */
std::string get_raw(const std::string& name);

/**
 * Return the value of the H2 environment variable name coerced to the
 * given type T via `from_string`.
 */
template <typename T>
inline T get(const std::string& name)
{
  return from_string<T>(get_raw(name));
}

namespace raw
{

/**
 * Return true if the environment variable name is set in the
 * environment.
 */
bool exists(const std::string& name);

/** Return the raw value of the environment variable name. */
std::string get_raw(const std::string& name);

/**
 * Return the environment variable name coerced to the given type T
 * via `from_string`.
 */
template <typename T>
inline T get(const std::string& name)
{
  return from_string<T>(get_raw(name));
}

}  // namespace raw

}  // namespace env

}  // namespace h2
