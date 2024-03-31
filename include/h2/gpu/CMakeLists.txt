################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/gpu")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Core COMPONENT CORE SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  error.hpp
  logger.hpp
  memory_utils.hpp
  runtime.hpp
  sync.hpp
  )

# Get other directories' files
if (H2_HAS_CUDA)
  add_subdirectory(cuda)
elseif (H2_HAS_ROCM)
  add_subdirectory(rocm)
endif ()
