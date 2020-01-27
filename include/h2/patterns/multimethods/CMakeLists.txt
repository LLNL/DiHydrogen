################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/multimethods")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Patterns COMPONENT PATTERNS SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  SwitchDispatcher.hpp
  )
