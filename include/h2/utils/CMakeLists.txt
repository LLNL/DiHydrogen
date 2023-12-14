################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Setup the base prefix for all the includes
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/utils")

h2_add_sources_to_target_and_install(
  TARGET H2Core COMPONENT CORE SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  As.hpp
  Cloneable.hpp
  Describable.hpp
  Error.hpp
  IntegerMath.hpp
  Logger.hpp
  )
