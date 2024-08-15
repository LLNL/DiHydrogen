################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# First add this directory's sources. CPP files aren't exported, so
# these are PRIVATE.
target_sources(H2Core PRIVATE Version.cpp)

# Subdirectories
add_subdirectory(core)
add_subdirectory(gpu)
add_subdirectory(tensor)
add_subdirectory(utils)
