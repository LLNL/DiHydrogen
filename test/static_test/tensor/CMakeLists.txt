################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(StaticTest
  PRIVATE
  fixed_size_tuple.cpp
  tensor_types.cpp
  strided_memory.cpp
)
