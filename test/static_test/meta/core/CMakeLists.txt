################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(StaticTest
  PRIVATE
  static_test_eq.cpp
  static_test_ifthenelse.cpp
  static_test_invocable.cpp
  static_test_lazy.cpp
  static_test_sfinae.cpp
  static_test_value_as_type.cpp
  )
