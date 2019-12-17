# @H2_LICENSE_TEXT@

#[=============[.rst
FindHalf
==========

Finds the Half half-precision library (FP16, IEEE-754 emulation).

The following variables will be defined::

  Half_FOUND          - True if the system has the Half library.
  Half_INCLUDE_DIRS   - The include directory needed for Half.
  Half_VERSION_STRING - The version of the Half library that was found
                        on the system.
  Half_VERSION        - A synonym for Half_VERSION_STRING.

The following cache variable will be set and marked as "advanced"::

  HALF_INCLUDE_DIR - The include directory needed for Half.
  HALF_HEADER_OK    - True if the found half library is usable.

In addition, the :prop_tgt:`IMPORTED` target ``Half::Half`` will
be created.

#]=============]

find_path(HALF_INCLUDE_DIR half.hpp
  HINTS ${HALF_DIR} $ENV{HALF_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_DIR
  DOC "The HALF header directory."
  )
find_path(HALF_INCLUDE_DIR half.hpp)

# Verify the header works
include(CheckCXXSourceCompiles)
set(_half_verify_code
  "#include <half.hpp>
int main(int, char**)
{
  half_float::half x
    = half_float::half_cast<half_float::half>(9.0);
}")
set(CMAKE_REQUIRED_INCLUDES ${HALF_INCLUDE_DIR})
check_cxx_source_compiles(
  "${_half_verify_code}" HALF_HEADER_OK)
set(CMAKE_REQUIRED_INCLUDES)

# Get the version out of it
if (HALF_INCLUDE_DIR)
  file(STRINGS "${HALF_INCLUDE_DIR}/half.hpp" HALF_VERSION_FILE_STRING
    REGEX "^// Version [0-9]+\.[0-9]+.*"
    LIMIT_COUNT 1)

  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
    HALF_VERSION_STRING
    "${HALF_VERSION_FILE_STRING}")
endif (HALF_INCLUDE_DIR)

if (HALF_INCLUDE_DIR)
  set(Half_INCLUDE_DIRS "${HALF_INCLUDE_DIR}")
endif ()
set(Half_VERSION_STRING "${HALF_VERSION_STRING}")
set(Half_VERSION "${Half_VERSION_STRING}")

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Half
  REQUIRED_VARS Half_INCLUDE_DIRS
  VERSION_VAR Half_VERSION)

# Setup the imported target
if (Half_FOUND)
  if (NOT TARGET Half::Half)
    add_library(Half::Half INTERFACE IMPORTED)
  endif (NOT TARGET Half::Half)

  # Set the include directories for the target
  set_property(TARGET Half::Half
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${Half_INCLUDE_DIRS})
endif (Half_FOUND)

mark_as_advanced(FORCE HALF_INCLUDE_DIR)
mark_as_advanced(FORCE HALF_HEADER_OK)
