################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(SeqCatchTests PRIVATE
  unit_test_allocator.cpp
  unit_test_dispatch.cpp
  unit_test_sync.cpp
  unit_test_types.cpp
  unit_test_version.cpp
)

if (H2_HAS_GPU)
  target_sources(GPUCatchTests PRIVATE
    unit_test_allocator.cpp
    unit_test_sync.cpp
  )
endif ()
