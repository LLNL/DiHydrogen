################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(StaticTest
  PRIVATE
  static_test_accessors.cpp
  static_test_append.cpp
  static_test_at.cpp
  static_test_cartprod.cpp
  static_test_expand.cpp
  static_test_find.cpp
  static_test_flatten.cpp
  static_test_length.cpp
  static_test_member.cpp
  static_test_reduce.cpp
  static_test_remove.cpp
  static_test_removeall.cpp
  static_test_replace.cpp
  static_test_replaceall.cpp
  static_test_select.cpp
  static_test_selectall.cpp
  static_test_sort.cpp
  static_test_unique.cpp
  static_test_zip.cpp
  )
