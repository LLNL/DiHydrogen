################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(H2Core PRIVATE
  base_utils.cpp
  copy.cpp)

if (H2_HAS_GPU)
  target_sources(H2Core PRIVATE
    copy.cu)
endif ()
