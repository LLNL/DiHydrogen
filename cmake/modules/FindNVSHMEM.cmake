################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Defines the following variables:
#   - NVSHMEM_FOUND
#   - NVSHMEM_LIBRARIES
#   - NVSHMEM_INCLUDE_DIRS
#
# Also creates an imported target NVSHMEM

message(STATUS "NVSHMEM_DIR: ${NVSHMEM_DIR}")

# Find the header
find_path(NVSHMEM_INCLUDE_DIRS nvshmem.h
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Directory with NVSHMEM header.")
find_path(NVSHMEM_INCLUDE_DIRS nvshmemx.h)

message(STATUS "NVSHMEM_INCLUDE_DIRS: ${NVSHMEM_INCLUDE_DIRS}")

# Find the library
find_library(NVSHMEM_LIBRARY nvshmem
  HINTS ${NVSHMEM_DIR} $ENV{NVSHMEM_DIR}
  PATH_SUFFIXES lib
  NO_DEFAULT_PATH
  DOC "The NVSHMEM library.")
find_library(NVSHMEM_LIBRARY nvshmem)

message(STATUS "NVSHMEM_LIBRARY: ${NVSHMEM_LIBRARY}")

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM
  DEFAULT_MSG
  NVSHMEM_LIBRARY NVSHMEM_INCLUDE_DIRS)

# Setup the imported target
if (NOT TARGET NVSHMEM::NVSHMEM)
  add_library(NVSHMEM::NVSHMEM INTERFACE IMPORTED)
endif (NOT TARGET NVSHMEM::NVSHMEM)

# Set the include directories for the target
set_property(TARGET NVSHMEM::NVSHMEM APPEND
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${NVSHMEM_INCLUDE_DIRS})

# Set the link libraries for the target
set_property(TARGET NVSHMEM::NVSHMEM APPEND
  PROPERTY INTERFACE_LINK_LIBRARIES ${NVSHMEM_LIBRARY})

set_property(TARGET NVSHMEM::NVSHMEM APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS -DNVSHMEM_TARGET)


#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE NVSHMEM_INCLUDE_DIRS)

# Set the libraries
set(NVSHMEM_LIBRARIES NVSHMEM::NVSHMEM)
mark_as_advanced(FORCE NVSHMEM_LIBRARY)

if (NVSHMEM_FOUND)
  set_property(TARGET cuda::toolkit APPEND PROPERTY
    INTERFACE_LINK_LIBRARIES ${CUDA_cudadevrt_LIBRARY})
  # Workaround for separable compilation with cooperative threading. see
  # https://stackoverflow.com/questions/53492528/cooperative-groupsthis-grid-causes-any-cuda-api-call-to-return-unknown-erro.
  # Adding this to INTERFACE_COMPILE_OPTIONS does not seem to solve the problem.
  # It seems that CMake does not add necessary options for device linking when cuda_add_executable/library is NOT used. See also
  # https://github.com/dealii/dealii/pull/5405
  string(APPEND CMAKE_CUDA_FLAGS "-gencode arch=compute_70,code=sm_70")
endif ()
