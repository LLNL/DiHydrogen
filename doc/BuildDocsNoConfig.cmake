################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# This is a script to build documentation without a proper
# configuration. The only difference should be that the
# "h2_config.hpp" header won't exist.
#
# A note from the CMake documentation: "If variables are defined using
# -D, this must be done before the -P argument."

# Required: Find Doxygen
find_program(DOXYGEN_PROGRAM doxygen)
if (NOT DOXYGEN_PROGRAM)
  message(FATAL_ERROR
    "Doxygen not found. Cannot build Doxygen documentation.")
endif ()

# Optional: Find Dot
find_program(DOT_PROGRAM dot)
if (DOT_PROGRAM)
  set(DOXYGEN_HAS_DOT TRUE)
else ()
  message("Warning: Dot program not found.")
endif ()

# Get the source root
get_filename_component(H2_SOURCE_ROOT "${CMAKE_CURRENT_LIST_DIR}" DIRECTORY)
set(H2_GENERATED_INCLUDE_DIR "${H2_SOURCE_ROOT}/include")
set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/doxygen_output")

macro(h2_make_yes_or_no VAR DEFAULT_VAL)
  if (NOT DEFINED ${VAR})
    set(${VAR} ${DEFAULT_VAL})
  endif ()

  if (${VAR})
    set(${VAR} YES)
  else ()
    set(${VAR} NO)
  endif ()
endmacro ()

h2_make_yes_or_no(DOXYGEN_GENERATE_HTML YES)
h2_make_yes_or_no(DOXYGEN_GENERATE_LATEX NO)
h2_make_yes_or_no(DOXYGEN_GENERATE_XML NO)

if (NOT (DOXYGEN_GENERATE_HTML OR DOXYGEN_GENERATE_LATEX
      OR DOXYGEN_GENERATE_HTML))
  message(FATAL_ERROR
    "Must set at least one of DOXYGEN_GENERATE_{HTML,LATEX,XML}=ON.")
endif ()

# Attempt to get the version
#
# TODO: Move to a proper module for use in regular CMake
find_program(GIT_PROGRAM git)
if (GIT_PROGRAM)
  execute_process(
    COMMAND ${GIT_PROGRAM} rev-parse --is-inside-work-tree
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE BUILDING_FROM_GIT_SOURCES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if (BUILDING_FROM_GIT_SOURCES)
    # Get the git version so that we can embed it into the executable
    execute_process(
      COMMAND ${GIT_PROGRAM} rev-parse --show-toplevel
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      OUTPUT_VARIABLE GIT_TOPLEVEL_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_PROGRAM} rev-parse --git-dir
      WORKING_DIRECTORY "${GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE GIT_GIT_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    execute_process(
      COMMAND ${GIT_PROGRAM} --git-dir "${GIT_GIT_DIR}" describe
      --abbrev=7 --always --dirty --tags
      WORKING_DIRECTORY "${GIT_TOPLEVEL_DIR}"
      OUTPUT_VARIABLE GIT_DESCRIBE_VERSION
      OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(H2_VERSION_STRING "Git version: ${GIT_DESCRIBE_VERSION}")

  endif (BUILDING_FROM_GIT_SOURCES)
endif (GIT_PROGRAM)

if (NOT H2_VERSION_STRING)
  set(H2_VERSION_STRING "Stable version: 0.0.1")
endif ()

configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/Doxyfile.in"
  "${CMAKE_CURRENT_LIST_DIR}/Doxyfile"
  @ONLY)

# Make the output directory
execute_process(
  COMMAND ${CMAKE_COMMAND} -E make_directory ${DOXYGEN_OUTPUT_DIR}
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  RESULT_VARIABLE MKDIR_SUCCESS)

if (NOT (MKDIR_SUCCESS EQUAL 0))
  message(FATAL_ERROR
    "Making \"${DOXYGEN_OUTPUT_DIR}\" failed with error message: "
    "${MKDIR_SUCCESS}")
endif ()

message(STATUS "Building Doxygen documentation.")

execute_process(
  COMMAND ${DOXYGEN_PROGRAM} "${CMAKE_CURRENT_LIST_DIR}/Doxyfile"
  WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
  OUTPUT_FILE "${DOXYGEN_OUTPUT_DIR}/DoxygenOutput.log"
  ERROR_FILE "${DOXYGEN_OUTPUT_DIR}/DoxygenError.log"
  RESULT_VARIABLE DOXYGEN_SUCCESS)


if (NOT (DOXYGEN_SUCCESS EQUAL 0))
  message(FATAL_ERROR
    "Running Doxygen failed with error message: ${DOXYGEN_SUCCESS}")
endif ()

message(STATUS "Doxygen documentation has been built in ${DOXYGEN_OUTPUT_DIR}")
message(STATUS "See ${DOXYGEN_OUTPUT_DIR}/DoxygenOutput.log and "
  "${DOXYGEN_OUTPUT_DIR}/DoxygenError.log for more details.")

if (DOXYGEN_GENERATE_HTML)
  message(STATUS "Visit ${DOXYGEN_OUTPUT_DIR}/html/index.html in a web browser "
    "to view documentation as Doxygen-generated HTML.")
endif ()

# TODO: Add sphinx support.
# TODO: Add "clean mode"
