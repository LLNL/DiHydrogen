################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

find_package(Doxygen 1.8.5)
find_package(Sphinx 1.8.0)

if (SPHINX_FOUND OR DOXYGEN_FOUND)
  add_custom_target(doc)
  add_custom_target(clean-doc)
endif ()

if (SPHINX_FOUND)
  add_subdirectory(sphinx)
endif ()

if (DOXYGEN_FOUND)
  set(H2_VERSION_STRING "DiHydrogen Version ${H2_VERSION}")
  set(H2_SOURCE_ROOT "${PROJECT_SOURCE_DIR}")
  set(H2_GENERATED_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")

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

  set(DOXYGEN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in
    ${PROJECT_SOURCE_DIR}/README.md
    )

  # Setup Dot support
  if (DOXYGEN_DOT_FOUND)
    set(DOXYGEN_HAVE_DOT "YES")
    get_filename_component(DOXYGEN_DOT_PATH "${DOXYGEN_DOT_EXECUTABLE}"
      DIRECTORY)
  else ()
    set(DOXYGEN_HAVE_DOT "NO")
  endif()

  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/doxygen")

  # Setup the doxyfile
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)

  # Apparently the Makefile generator doesn't do this properly
  add_custom_target(doxygen-make-dir
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DOXYGEN_OUTPUT_DIR}"
    COMMENT "Creating the doxygen output directory."
    VERBATIM)

  add_custom_target(doxygen-generate-html
    COMMAND Doxygen::doxygen "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    BYPRODUCTS "${DOXYGEN_OUTPUT_DIR}/html/index.html"
    WORKING_DIRECTORY "${PROJECT_BINARY_DIR}"
    SOURCES "${DOXYGEN_SOURCES}"  "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
    DEPENDS doxygen-make-dir
    COMMENT
    "Generating Doxygen documentation."
    VERBATIM)

  add_custom_target(doxygen
    COMMAND ${CMAKE_COMMAND} -E echo
    " --> Doxygen documentation built at ${DOXYGEN_OUTPUT_DIR}/html/index.html"
    DEPENDS doxygen-generate-html
    COMMENT
    "Doxygen documentation built at ${DOXYGEN_OUTPUT_DIR}/html/index.html"
    VERBATIM)

  add_custom_target(clean-doxygen
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${DOXYGEN_OUTPUT_DIR}
    COMMENT "Removing Doxygen documentation."
    VERBATIM)

  add_dependencies(doc doxygen)
  add_dependencies(clean-doc clean-doxygen)

  message(STATUS
    "Added target \"doxygen\" for generating Doxygen documentation.")
endif()
