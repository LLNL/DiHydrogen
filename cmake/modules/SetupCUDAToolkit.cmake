################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Copied from LBANN

find_package(CUDA REQUIRED)
if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 14)
endif ()
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)

find_package(NVTX REQUIRED)

if (NOT TARGET cuda::toolkit)
  add_library(cuda::toolkit INTERFACE IMPORTED)
endif ()

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:
  -g --expt-extended-lambda>)

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--device-debug>)

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES ${CUDA_LIBRARIES} cuda cuda::nvtx)

# Set the target PTX and architecture version. If the auto detection is successful, CMake should
# return a list of length 2, e.., "-gencode:arch=compute_62,code=sm_62". Otherwise, it should enumerate
# all the version numbers, however, adding multiple '-gencode' options with set_property does not seem
# work. More fundamentally, it's not desirable to generate ptx and binary codes for all versions since
# that would inflate compilation time. The code below checks whether automatic detection is successful,
# and only then add the options to the nvcc command-line option list.

cuda_select_nvcc_arch_flags(ARCH_FLAGS Auto)

list(LENGTH ARCH_FLAGS arch_flag_length)
if (${arch_flag_length} GREATER 2)
  message(STATUS "Ignoring ${ARCH_FLAGS}")
else ()
  message(STATUS "Targeting CUDA architecture version ${ARCH_FLAGS_readable}")
  set_property(TARGET cuda::toolkit APPEND PROPERTY
    INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
endif ()
