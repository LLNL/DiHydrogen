################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Proper C++ files to add to the library
if (H2_HAS_CUDA)
  set(_GPU_DIR "cuda")
elseif (H2_HAS_ROCM)
  set(_GPU_DIR "rocm")
endif ()

target_sources(H2Core PRIVATE
  logger.cpp
  memory_utils.cpp
  ${_GPU_DIR}/runtime.cpp
  )
