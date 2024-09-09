################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
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
  const_for.hpp
  Describable.hpp
  environment_vars.hpp
  Error.hpp
  function_traits.hpp
  IntegerMath.hpp
  Logger.hpp
  passkey.hpp
  strings.hpp
  tuple_utils.hpp
  typename.hpp
  unique_ptr_cast.hpp
  )
