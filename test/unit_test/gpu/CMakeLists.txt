################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

if (H2_HAS_GPU)
  target_sources(GPUCatchTests PRIVATE
    unit_test_launch_kernel.cpp
    test_kernel.cu
  )
endif ()
