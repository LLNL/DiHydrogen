################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

################################################################
# Initialize RPATH (always full RPATH)
# Note: see https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# Use RPATH on OS X
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif ()

# Use (i.e. don't skip) RPATH for build
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Use same RPATH for build and install
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" _IS_SYSTEM_DIR)
if (${_IS_SYSTEM_DIR} STREQUAL "-1")
  # Set the install RPATH correctly
  list(APPEND CMAKE_INSTALL_RPATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif ()
