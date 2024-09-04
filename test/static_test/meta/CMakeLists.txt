################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

add_subdirectory(core)
add_subdirectory(partial_functions)
add_subdirectory(typelist)

target_sources(StaticTest
  PRIVATE
  static_test_core.cpp
  )
