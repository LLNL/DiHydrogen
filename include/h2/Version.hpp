////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

/** @namespace h2
 *  @brief The main namespace for DiHydrogen.
 */

namespace h2
{
/** @brief Get the version string for DiHydrogen
 *  @returns A string of the format "MAJOR.MINOR.PATCH".
 */
std::string Version() noexcept;

} // namespace h2
