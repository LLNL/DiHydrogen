# @H2_LICENSE_TEXT@

#[=============[.rst
FindBreathe
===========

Finds the Breathe ``breathe-apidoc`` tool.

The following variables will be defined::

  Breathe_FOUND          - True if the system has the Breathe
                           breathe-apidoc tool.
  Breathe_EXECUTABLE     - The breathe-apidoc executable found on the system.
  Breathe_VERSION_STRING - The version of the breathe-apidoc tool that
                           was found on the system.
  Breathe_VERSION        - A synonym for Breathe_VERSION_STRING.

The following cache variable will be set and marked as "advanced"::

  BREATHE_APIDOC_PROGRAM - The breathe-apidoc executable found on the system.

In addition, the :prop_tgt:`IMPORTED` target ``Breathe::Breathe`` will
be created.

#]=============]

find_program(BREATHE_APIDOC_PROGRAM breathe-apidoc
  HINTS ${BREATHE_DIR} $ENV{BREATHE_DIR}
  ${SPHINX_DIR} $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
  DOC "The breathe documentation tool."
  NO_DEFAULT_PATH)
find_program(BREATHE_APIDOC_PROGRAM breathe-apidoc)

if (BREATHE_APIDOC_PROGRAM)
  execute_process(COMMAND ${BREATHE_APIDOC_EXECUTABLE} --version
    OUTPUT_VARIABLE BREATHE_APIDOC_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
    BREATHE_APIDOC_VERSION
    "${BREATHE_APIDOC_VERSION_STRING}")
endif ()

set(Breathe_EXECUTABLE ${BREATHE_APIDOC_PROGRAM})
set(Breathe_VERSION_STRING ${BREATHE_APIDOC_VERSION})
set(Breathe_VERSION ${Breathe_VERSION_STRING})

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Breathe
  REQUIRED_VARS Breathe_EXECUTABLE
  VERSION_VAR Breath_VERSION)

if (NOT TARGET Breathe::Breathe)
  add_executable(Breathe::Breathe IMPORTED)
  set_property(TARGET Breathe::Breathe
    PROPERTY IMPORTED_LOCATION "${Breathe_EXECUTABLE}")
endif (NOT TARGET Breathe::Breathe)

mark_as_advanced(FORCE BREATHE_APIDOC_PROGRAM)
