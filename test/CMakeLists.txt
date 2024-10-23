################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

if (H2_ENABLE_TESTS)
  # Setup the Catch2 stuff.
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG fa43b77429ba76c462b1898d6cd2f2d7a9416b14 # v3.7.1
    FIND_PACKAGE_ARGS 3.0.0 CONFIG
  )
  FetchContent_MakeAvailable(Catch2)
  report_dep(Catch2)

  # Add the actual drivers
  add_subdirectory(unit_test)
endif ()

# Static test targets are always added
add_subdirectory(static_test)
