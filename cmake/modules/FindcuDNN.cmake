################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

#[=============[.rst
FindcuDNN
==========

Finds the cuDNN library.

The following variables will be defined::

  cuDNN_FOUND          - True if the system has the cuDNN library.
  cuDNN_INCLUDE_DIRS   - The include directory needed for cuDNN.
  cuDNN_LIBRARIES      - The libraries needed for cuDNN.
  cuDNN_VERSION        - The version for cuDNN.

The following cache variable will be set and marked as "advanced"::

  cuDNN_INCLUDE_PATH - The include directory needed for cuDNN.
  cuDNN_LIBRARY      - The library needed for cuDNN.

In addition, the :prop_tgt:`IMPORTED` target ``h2::cuDNN`` will
be created.

#]=============]

find_path(cuDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cudnn header."
  )
find_path(cuDNN_INCLUDE_PATH cudnn.h)

find_library(cuDNN_LIBRARY cudnn
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The cudnn library."
  )
find_library(cuDNN_LIBRARY cudnn)

# Shamelessly copied from LBANN
set(cuDNN_VERSION)
if (cuDNN_INCLUDE_PATH)
  set(_cuDNN_VERSION_SRC "
#include <stdio.h>
#include <cudnn_version.h>
int main() {
  printf(\"%d.%d.%d\\n\", CUDNN_MAJOR, CUDNN_MINOR, CUDNN_PATCHLEVEL);
  return 0;
}
")

  file(
    WRITE
    "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx"
    "${_cuDNN_VERSION_SRC}\n")

  try_run(
    _cuDNN_RUN_RESULT _cuDNN_COMPILE_RESULT
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/src.cxx
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${cuDNN_INCLUDE_PATH}"
    RUN_OUTPUT_VARIABLE cuDNN_VERSION
    COMPILE_OUTPUT_VARIABLE _cuDNN_COMPILE_OUTPUT)
endif ()

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
  DEFAULT_MSG cuDNN_VERSION cuDNN_LIBRARY cuDNN_INCLUDE_PATH)

if (NOT TARGET h2::cuDNN)
  add_library(h2::cuDNN INTERFACE IMPORTED)
endif (NOT TARGET h2::cuDNN)

target_include_directories(h2::cuDNN INTERFACE "${cuDNN_INCLUDE_PATH}")
target_link_libraries(h2::cuDNN INTERFACE "${cuDNN_LIBRARY}")

set(cuDNN_INCLUDE_DIRS "${cuDNN_INCLUDE_PATH}")
set(cuDNN_LIBRARIES h2::cuDNN)

mark_as_advanced(FORCE
  cuDNN_INCLUDE_PATH
  cuDNN_LIBRARY)
