################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

macro (h2_append_full_path VAR)
  foreach (filename ${ARGN})
    list(APPEND ${VAR} "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach ()
endmacro ()

# A handy macro to add the current source directory to a local
# filename. To be used for creating a list of sources.
macro (h2_set_full_path VAR)
  unset(__tmp_names)
  h2_append_full_path(__tmp_names ${ARGN})
  set(${VAR} "${__tmp_names}")
endmacro()

macro (h2_set_default VAR DEFAULT_VALUE)
  if (NOT ${VAR})
    set(${VAR} ${DEFAULT_VALUE})
  endif ()
endmacro ()

macro (h2_assert_value VAR)
  if (NOT ${VAR})
    message(FATAL_ERROR "Variable ${VAR} requires a value.")
  endif ()
endmacro ()

# Need a macro that will iterate through a list of sources and add
# each to a target, with complete build/install interface
# support. This will also install the file to the suitable location.
macro (h2_add_sources_to_target_and_install)
  set(_H2_OPTIONS)
  set(_H2_SINGLE_VAL_ARGS TARGET COMPONENT SCOPE INSTALL_PREFIX)
  set(_H2_MULTI_VAL_ARGS SOURCES)

  cmake_parse_arguments(
    _H2_INTERNAL
    "${_H2_OPTIONS}"
    "${_H2_SINGLE_VAL_ARGS}"
    "${_H2_MULTI_VAL_ARGS}"
    ${ARGN})

  # Verify that we have enough information to proceed.
  h2_assert_value(_H2_INTERNAL_INSTALL_PREFIX)
  h2_assert_value(_H2_INTERNAL_TARGET)
  h2_assert_value(_H2_INTERNAL_SOURCES)

  # Handle defaults if nothing
  h2_set_default(_H2_INTERNAL_SCOPE INTERFACE)
  h2_set_default(_H2_INTERNAL_COMPONENT Unspecified)

  # Add all the sources to the target.
  foreach (src IN LISTS _H2_INTERNAL_SOURCES)
    target_sources(${_H2_INTERNAL_TARGET} ${_H2_INTERNAL_SCOPE}
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/${src}>
      $<INSTALL_INTERFACE:${_H2_INTERNAL_INSTALL_PREFIX}/${src}>
      )
  endforeach ()

  # Add all the sources to the install target.
  install(
    FILES       ${_H2_INTERNAL_SOURCES}
    DESTINATION ${_H2_INTERNAL_INSTALL_PREFIX}
    COMPONENT   ${_H2_INTERNAL_COMPONENT}
    )

endmacro ()
