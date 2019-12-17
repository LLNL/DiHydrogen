# @H2_LICENSE_TEXT@

# Setup the Catch2 stuff.
find_package(Catch2 2.0.0 CONFIG QUIET
  HINTS ${CATCH2_DIR} $ENV{CATCH2_DIR} ${CATCH_DIR} $ENV{CATCH_DIR}
  PATH_SUFFIXES lib64/cmake/Catch2 lib/cmake/Catch2
  NO_DEFAULT_PATH)
if (NOT Catch2_FOUND)
  find_package(Catch2 2.0.0 CONFIG QUIET REQUIRED)
endif ()

message(STATUS "Found Catch2: ${Catch2_DIR}")
include(Catch)

# Add the actual drivers
add_subdirectory(unit_test)
