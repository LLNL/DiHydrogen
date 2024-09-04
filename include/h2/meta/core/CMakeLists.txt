################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/core")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Meta COMPONENT META SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Eq.hpp
  IfThenElse.hpp
  Invocable.hpp
  Lazy.hpp
  SFINAE.hpp
  ValueAsType.hpp
  )
