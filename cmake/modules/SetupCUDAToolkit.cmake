################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

find_package(CUDAToolkit 11.0.0 REQUIRED)
if (H2_ENABLE_DISTCONV_LEGACY)
  find_package(cuDNN REQUIRED)
endif ()

if (NOT TARGET h2::cuda_toolkit)
  add_library(h2::cuda_toolkit INTERFACE IMPORTED)
endif ()

target_compile_options(
  h2::cuda_toolkit
  INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:-g>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CONFIG:Debug>>:--device-debug>)

target_include_directories(
  h2::cuda_toolkit
  INTERFACE
  "${CUDA_INCLUDE_DIRS}")

target_link_libraries(
  h2::cuda_toolkit
  INTERFACE
  $<TARGET_NAME_IF_EXISTS:cuda::cudnn>
  CUDA::nvToolsExt
  CUDA::nvml
  CUDA::cuda_driver
  CUDA::cudart)

# Arch flags are now set automatically. Be sure to set
# CMAKE_CUDA_ARCHITECTURES on the command line.
