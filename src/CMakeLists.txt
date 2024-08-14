################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# First add this directory's sources. CPP files aren't exported, so
# these are PRIVATE.
target_sources(H2Core PRIVATE Version.cpp)

# Subdirectories
add_subdirectory(core)
add_subdirectory(gpu)
add_subdirectory(tensor)
add_subdirectory(utils)

# If using ROCm, set the appropriate property for all .cu files.
if (H2_HAS_GPU AND H2_ENABLE_ROCM)
  get_target_property(CORE_SOURCES H2Core SOURCES)
  foreach (SOURCE_FILE ${CORE_SOURCES})
    if (SOURCE_FILE MATCHES "\\.(cu|cuh)$")
      set_source_file_properties(${SOURCE_FILE} PROPERTIES LANGUAGE HIP)
    endif ()
  endforeach ()
endif ()
