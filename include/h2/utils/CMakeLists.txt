################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Setup the base prefix for all the includes
set(H2_CURRENT_INSTALL_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/utils")

h2_add_sources_to_target_and_install(
  TARGET H2Core COMPONENT CORE SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Error.hpp
  Logger.hpp
  )

# FIXME(KLG): Should this go here?
target_link_libraries(H2Core PRIVATE spdlog::spdlog)
# FIXME(KLG): Why is this stupid? Can't find spdlog in header
target_include_directories(H2Core PUBLIC
  ${spdlogDIR}
  /g/g13/graham63/workspace/code/pascal/H2_logging/spdlog-install/usr/local/include)
