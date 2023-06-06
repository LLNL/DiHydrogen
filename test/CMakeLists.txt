################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

if (H2_ENABLE_TESTS)
  # Setup the Catch2 stuff.
  find_package(Catch2 3.0.0 CONFIG REQUIRED)

  message(STATUS "Found Catch2: ${Catch2_DIR}")
  message(STATUS "Catch2 Version: ${Catch2_VERSION}")
  include(Catch)

  # Add the actual drivers
  add_subdirectory(unit_test)
endif ()

# Static test targets are always added
add_subdirectory(static_test)
