################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(SeqCatchTests PRIVATE
  unit_test_cub_allocator.cpp
  unit_test_malloc_allocator.cpp
  unit_test_mallocasync_allocator.cpp
  unit_test_pool_allocator.cpp
  )
