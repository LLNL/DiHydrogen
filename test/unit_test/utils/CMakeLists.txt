################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(SeqCatchTests PRIVATE
  unit_test_as.cpp
  unit_test_cloneable.cpp
  unit_test_describable.cpp
  unit_test_environment_vars.cpp
  unit_test_error.cpp
  unit_test_integer_math.cpp
  unit_test_logging.cpp
  unit_test_strings.cpp
  unit_test_typename.cpp
  unit_test_unique_ptr_cast.cpp
  )
