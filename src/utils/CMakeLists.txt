################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Proper C++ files to add to the library
target_sources(H2Core PRIVATE
  environment_vars.cpp
  Error.cpp
  Logger.cpp
  typename.cpp
  )
