################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Proper C++ files to add to the library
target_sources(H2Core PRIVATE
  Error.cpp
  Logger.cpp
  )
