################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
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
  dist_tensor_base.hpp
  dist_tensor.hpp
  dist_types.hpp
  dist_utils.hpp
  fixed_size_tuple.hpp
  hydrogen_interop.hpp
  proc_grid.hpp
  raw_buffer.hpp
  strided_memory.hpp
  tensor_base.hpp
  tensor_types.hpp
  tensor_utils.hpp
  tensor.hpp
  tuple_utils.hpp
  )
