////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2_config.hpp>

#include <h2/Version.hpp>

#define STRINGIFY(thing) LAYER_TWO(thing)
#define LAYER_TWO(thing) #thing

namespace h2
{
std::string Version() noexcept
{
  return STRINGIFY(H2_VERSION_MAJOR) "." STRINGIFY(
    H2_VERSION_MINOR) "." STRINGIFY(H2_VERSION_PATCH);
}

}  // namespace h2
