# @H2_LICENSE_TEXT@

# A handy macro to add the current source directory to a local
# filename. To be used for creating a list of sources.
macro (h2_set_full_path VAR)
  unset(__tmp_names)
  foreach (filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

macro (tom_install_with_prefix BASE_DIR INSTALL_BASE_DIR)
  file(RELATIVE_PATH __this_dir_rel_path
    "${BASE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
  install(FILES ${ARGN}
    DESTINATION "${INSTALL_BASE_DIR}/${__this_dir_rel_path}"
    COMPONENT ${TOM_CURRENT_INSTALL_COMPONENT})
endmacro ()

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
