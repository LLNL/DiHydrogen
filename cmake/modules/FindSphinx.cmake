# @H2_LICENSE_TEXT@

#[=============[.rst
FindSphinx
===========

Finds the Sphinx ``sphinx-build`` tool.

The following variables will be defined::

  Sphinx_FOUND          - True if the system has the Sphinx
                           sphinx-build tool.
  Sphinx_EXECUTABLE     - The sphinx-build executable found on the system.
  Sphinx_VERSION_STRING - The version of the sphinx-build tool that
                           was found on the system.
  Sphinx_VERSION        - A synonym for Sphinx_VERSION_STRING.

The following cache variable will be set and marked as "advanced"::

  SPHINX_BUILD_PROGRAM - The sphinx-build executable found on the system.

In addition, the :prop_tgt:`IMPORTED` target ``Sphinx::Sphinx`` will
be created.

#]=============]

find_program(SPHINX_BUILD_PROGRAM sphinx-build
  HINTS ${SPHINX_DIR} $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
  DOC "The sphinx-build documentation tool."
  NO_DEFAULT_PATH)
find_program(SPHINX_BUILD_PROGRAM sphinx-build)

if (SPHINX_BUILD_PROGRAM)
  execute_process(COMMAND ${SPHINX_BUILD_PROGRAM} --version
    OUTPUT_VARIABLE SPHINX_BUILD_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
    SPHINX_BUILD_VERSION
    "${SPHINX_VERSION_STRING}")
endif ()

set(Sphinx_EXECUTABLE ${SPHINX_BUILD_PROGRAM})
set(Sphinx_VERSION_STRING ${SPHINX_BUILD_VERSION})
set(Sphinx_VERSION ${Sphinx_VERSION_STRING})

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Sphinx
  REQUIRED_VARS Sphinx_EXECUTABLE
  VERSION_VAR Sphinx_VERSION)

if (Sphinx_FOUND)
  if (NOT TARGET Sphinx::Sphinx)
    add_executable(Sphinx::Sphinx IMPORTED)
  endif (NOT TARGET Sphinx::Sphinx)
  set_property(TARGET Sphinx::Sphinx
    PROPERTY IMPORTED_LOCATION "${Sphinx_EXECUTABLE}")
endif (Sphinx_FOUND)

mark_as_advanced(FORCE SPHINX_BUILD_PROGRAM)
