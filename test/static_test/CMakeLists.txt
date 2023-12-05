################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# This library might be empty; definitely shouldn't provide useful
# symbols -- it's really just a way to "run the tests" for static
# testing. As such, it is not installed and it is not exported in any
# way.
add_library(StaticTest MODULE "")
add_subdirectory(meta)
add_subdirectory(tensor)
add_subdirectory(utils)
target_link_libraries(StaticTest PRIVATE H2Meta H2Core)
set_target_properties(StaticTest
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)
