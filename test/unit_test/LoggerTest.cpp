////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////

#include <h2/utils/Logger.hpp>

#include <iostream>

int main(int, char*[])
{
  h2::Logger logger;

  H2_TRACE("Why don't the first two work?");

  H2_DEBUG("Don't show this one to users");

  H2_INFO("testing spdlog v{}.{}", 0, 1);

  H2_ERROR("You've encountered error {}", 32)

  H2_WARN("Easy padding in numbers like {:08d}", 12);

  H2_CRITICAL("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);

  return 0;
}
