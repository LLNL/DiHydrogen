################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(StaticTest
  PRIVATE
  static_test_cloneable.cpp
  static_test_function_traits.cpp
  static_test_tuple_utils.cpp
  )
