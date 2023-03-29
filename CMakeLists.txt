################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

cmake_minimum_required(VERSION 3.21.0)

project(DiHydrogen
  VERSION 0.3.0
  DESCRIPTION "A distributed multilinear algebra interface library."
  HOMEPAGE_URL "https://github.com/llnl/DiHydrogen"
  LANGUAGES CXX)

# Add CMake modules
set(CMAKE_MODULE_DIR "${CMAKE_SOURCE_DIR}/cmake/modules")
set(CONFIG_FILE_DIR "${CMAKE_SOURCE_DIR}/cmake/config")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_DIR})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RELEASE CACHE STRING "CMake build type" FORCE)
endif ()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE)

#
# It will be treated as a FATAL_ERROR to enable any feature and then
# subsequently fail to find its dependencies.
#

include(CMakeDependentOption)

option(H2_ENABLE_DISTCONV_LEGACY
  "Enable the legacy DistConv code branch."
  OFF)

option(H2_ENABLE_CUDA
  "Use the CUDA backend for DistConv features of DiHydrogen."
  OFF)

option(H2_ENABLE_HIP_ROCM
  "Search for and enable ROCm/HIP language features in DiHydrogen."
  OFF)

option(H2_ENABLE_ROCM
  "Search for and enable ROCm/HIP language features in DiHydrogen."
  ${H2_ENABLE_HIP_ROCM})

# We need only one of CUDA or ROCm
if (H2_ENABLE_CUDA AND H2_ENABLE_ROCM)
  message(FATAL_ERROR "Must enable no more than one of CUDA or ROCm.")
endif ()

# Just in case I missed anything
if (H2_ENABLE_ROCM AND NOT H2_ENABLE_HIP_ROCM)
  set(H2_ENABLE_HIP_ROCM ON CACHE BOOL "Use HIP/ROCm backend." FORCE)
endif ()

# Sanity-check the arguments.
if (H2_ENABLE_DISTCONV_LEGACY)
  # We need one of CUDA or ROCm
  if (NOT (H2_ENABLE_CUDA OR H2_ENABLE_ROCM))
    message(FATAL_ERROR "Must enable one of CUDA and ROCm with DistConv.")
  endif ()

  # These might become full-fledged options later, but I hope
  # not. Aluminum should just be the communication interface that H2
  # uses across the board.
  set(H2_ENABLE_ALUMINUM ${H2_ENABLE_DISTCONV_LEGACY}
    CACHE BOOL "Use the Aluminum library for communications.")

  set(H2_ENABLE_OPENMP ${H2_ENABLE_DISTCONV_LEGACY}
    CACHE BOOL "Enable CPU acceleration with OpenMP threads.")
endif ()

option(H2_DEVELOPER_BUILD
  "Enable extra warnings and force tests to be enabled."
  OFF)

option(H2_ENABLE_CODE_COVERAGE
  "Enable code coverage instrumentation. Requires compiler support."
  OFF)

# This is exposed as an option in case Catch2 isn't available or
# whatever.
option(H2_ENABLE_TESTS
  "Enable test codes. Requires Catch2."
  ${H2_DEVELOPER_BUILD})

option(H2_ENABLE_WERROR
  "Enable the \"-Werror\" flag. Requires compiler support."
  OFF)

# Hack
set(MPI_ASSUME_NO_BUILTIN_MPI ON
  CACHE BOOL
  "Force compiler interrogation for MPI")

include(GNUInstallDirs)
include(H2CMakeUtils)
include(H2CXXFeatureDetection)

include(SetupCXX) # This just sets RPATH settings

# There are a variety of ways to inject flags into C++ compilers, and
# some behavioral variations across compilers. I don't want to care
# about all these intricate data, so unless we know we're in the
# comfortable realm of Clang or GCC, we don't touch the flags at all
# -- users are free to modify them as needed by the usual CMake
# mechanisms for adjusting language compiler flags.
#
# In the Clang/GNU universe, flags are read left to right, so
# appending flags allows a user to override any flags that may have
# been automatically injected. Thus, we gather our "developer" flags
# into a variable and prepend that string to the usual CMakery. Users,
# then, should continue to add custom flags via the usual CMake
# methods for doing so and these options should not interfere with
# those mechanisms. That is, a user should NOT mess with
# H2_CMAKE_CXX_FLAGS.
if (CMAKE_CXX_COMPILER_ID MATCHES ".*Clang"
    OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")

  list(APPEND H2_CMAKE_CXX_FLAGS -g3 -ggdb3)
  if (H2_DEVELOPER_BUILD)
    list(APPEND H2_CMAKE_CXX_FLAGS -Wall -Wextra -pedantic
      -Wno-unused-parameter -Wnon-virtual-dtor -Wshadow)
  endif ()
  if (H2_ENABLE_WERROR)
    list(APPEND H2_CMAKE_CXX_FLAGS -Werror)
  endif ()

  # PREPEND the options.
  set(CMAKE_CXX_FLAGS "${H2_CMAKE_CXX_FLAGS_STR} ${CMAKE_CXX_FLAGS}")

else ()

  message(NOTICE
    "You are using a compiler that CMake doesn't recognize as a Clang "
    "or GNU compiler. This should be fine, but this build system will "
    "not add any additional flags beyond CMake's defaults for you. "
    "You must add any additional flags manually using the usual CMake "
    "mechanisms (e.g., CMAKE_CXX_FLAGS). CMake will (attempt to) use"
    "an appropriate C++ standard (C++17 at time of writing) via the "
    "cxx_std_17 compiler feature. If your compiler does not support "
    "this feature, another warning will appear later.")

endif ()

if (H2_ENABLE_DISTCONV_LEGACY)
  include(SetupMPI)
endif ()

# C++ features

# TODO: h2_cxx_set_cxx_standard(H2_CXX_STANDARD)
h2_cxx_determine_pretty_function(H2_PRETTY_FUNCTION)
h2_cxx_determine_restrict_qualifier(H2_RESTRICT)

if (H2_ENABLE_OPENMP)
  find_package(OpenMP 3.0 REQUIRED COMPONENTS CXX)
  set(H2_HAS_OPENMP ${OPENMP_CXX_FOUND})
endif ()

if (H2_ENABLE_FP16)
  find_package(Half REQUIRED)
  set(H2_HAS_HALF ${HALF_FOUND})
endif ()

if (H2_DEVELOPER_BUILD)
  # It is *not* a FATAL_ERROR to not have clang-format.
  include(H2AddClangFormat)
endif ()

if (H2_ENABLE_CODE_COVERAGE)
  include(H2CXXCodeCoverage)
  add_custom_target(coverage)
  set(H2_CODE_COVERAGE TRUE)
endif ()

# Required dependencies

if (H2_ENABLE_CUDA)
  enable_language(CUDA)
  include(SetupCUDAToolkit)
  if (CMAKE_CUDA_COMPILER)
    set(H2_HAS_CUDA TRUE)
  endif ()
  find_package(NVSHMEM)

  set(H2_CUDA_LIBS
    h2::cuda_toolkit)
  if (NVSHMEM_FOUND)
    list(APPEND H2_CUDA_LIBS NVSHMEM::NVSHMEM)
  endif ()
endif (H2_ENABLE_CUDA)

find_package(spdlog REQUIRED)
set(H2_HAS_SPDLOG TRUE)

# This is a placeholder.
if (H2_ENABLE_ROCM)
  find_package(hip CONFIG REQUIRED) # This must be first.

  # This is used to compile .cu files as HIP files.
  enable_language(HIP)

  find_library(HSA_LIBRARY hsa-runtime64
    HINTS ${ROCM_PATH}/hsa $ENV{ROCM_PATH}/hsa
    PATH_SUFFIXES lib lib64
    DOC "HSA runtime library"
    NO_DEFAULT_PATH)
  find_library(HSA_LIBRARY hsa-runtime64)
  if (NOT HSA_LIBRARY)
    message(FATAL_ERROR "HSA Runtime library not found.")
  endif ()

  # These should be part of a bundled ROCm release.
  find_package(MIOpen CONFIG REQUIRED)
  find_package(hipcub CONFIG REQUIRED)
  find_package(rocm_smi CONFIG REQUIRED)

  find_package(Roctracer MODULE REQUIRED)

  set(H2_ROCM_LIBS
    hip::host
    hip::hipcub
    MIOpen
    rocm_smi64
    ${Roctracer_LIBRARIES}
    ${HSA_LIBRARY})
  set(H2_HAS_ROCM TRUE)
endif ()

if (H2_HAS_CUDA OR H2_HAS_ROCM)
  set(H2_HAS_GPU TRUE)
endif ()

# DiHydrogen will use MPI-3 features extensively. Until proven
# otherwise, assume that the CMake module version checking is
# accurate.
#
set(MPI_ASSUME_NO_BUILTIN_MPI ON)
if (H2_ENABLE_DISTCONV_LEGACY)
  find_package(MPI 3.0.0 COMPONENTS CXX REQUIRED)

  get_target_property(
    __mpi_compile_options MPI::MPI_CXX INTERFACE_COMPILE_OPTIONS)
  if (__mpi_compile_options)
    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_COMPILE_OPTIONS
      $<$<COMPILE_LANGUAGE:CXX>:${__mpi_compile_options}>)
    unset(__mpi_compile_options)
  endif ()
endif ()

if (H2_ENABLE_ALUMINUM)
  find_package(Aluminum 1.0.0 CONFIG QUIET)
  if (NOT Aluminum_FOUND AND Aluminum_NOT_FOUND_MESSAGE)
    message(STATUS
      "A candidate Aluminum > v1.0.0 was found, but was not selected:")
    message(STATUS
      "  ${Aluminum_NOT_FOUND_MESSAGE}")
  endif ()
  # Try again, since we're technically ok with >v0.7.0
  if (NOT Aluminum_FOUND)
    find_package(Aluminum 0.7.0 CONFIG QUIET)
    if (NOT Aluminum_FOUND AND Aluminum_NOT_FOUND_MESSAGE)
      message(STATUS
        "A candidate Aluminum > v0.7.0 was found, but was not selected:")
      message(STATUS
        "  ${Aluminum_NOT_FOUND_MESSAGE}")
    endif ()
  endif ()
  if (NOT Aluminum_FOUND)
    message(FATAL_ERROR "Aluminum support required but not found.")
  endif ()
  if (H2_ENABLE_DISTCONV_LEGACY AND NOT AL_HAS_NCCL)
    message(FATAL_ERROR
      "DistConv requires Aluminum with at least the NCCLBackend enabled.")
  endif ()
endif ()

if (MPI_FOUND AND H2_HAS_ROCM)
  # Fix things in the MPI::MPI_CXX target
  get_target_property(_MPI_CXX_INCL_DIRS
    MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
  if (NOT _MPI_CXX_INCL_DIRS)
    set(_MPI_CXX_INCL_DIRS ${MPI_CXX_COMPILER_INCLUDE_DIRS})
  endif ()

  if ((NOT _MPI_CXX_INCL_DIRS)
      AND (DEFINED ENV{PE_ENV})
      AND ("${MPI_CXX_COMPILER}" STREQUAL "${CMAKE_CXX_COMPILER}"))

    execute_process(
      COMMAND "${CMAKE_CXX_COMPILER}" "--cray-print-opts=cflags"
      OUTPUT_VARIABLE _CRAY_INCLUDE_PATHS)
    if (_CRAY_INCLUDE_PATHS)
      string(STRIP "${_CRAY_INCLUDE_PATHS}" _CRAY_INCLUDE_PATHS_STRIPPED)
      string(REPLACE "-I" "" _CRAY_INCLUDE_PATHS_NO_I
        "${_CRAY_INCLUDE_PATHS_STRIPPED}")
      string(REPLACE " " ";" _MPI_CXX_INCL_DIRS
        "${_CRAY_INCLUDE_PATHS_NO_I}")
    endif ()
  endif ()

  if (_MPI_CXX_INCL_DIRS)
    target_include_directories(MPI::MPI_CXX
      INTERFACE
      $<$<COMPILE_LANGUAGE:HIP>:${_MPI_CXX_INCL_DIRS}>)
  endif ()
endif ()

if (MPI_CXX_FOUND)
  message(STATUS "H2 HAS MPI_CXX")
  set(H2_HAS_MPI TRUE)
else ()
  set(H2_HAS_MPI FALSE)
endif ()

# Decide where to put generated include files.
set(CMAKE_GENERATED_INCLUDE_DIRECTORY
  "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_INCLUDEDIR}")

# Decide where to put built executable files.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  "${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}")

# Build the library
configure_file(
  "${CONFIG_FILE_DIR}/h2_config.hpp.in"
  "${CMAKE_GENERATED_INCLUDE_DIRECTORY}/h2_config.hpp"
  @ONLY)
install(
  FILES "${CMAKE_GENERATED_INCLUDE_DIRECTORY}/h2_config.hpp"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  COMPONENT CORE)

# Add a target to collect things that are fundamental to the library,
# such as the version.
add_library(H2Core
  "${CMAKE_GENERATED_INCLUDE_DIRECTORY}/h2_config.hpp")

add_subdirectory(include/h2)
add_subdirectory(src)

target_include_directories(
  H2Core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if (NOT CMAKE_CXX_STANDARD)
  set_target_properties(H2Core
    PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON)
endif ()

# Compiler flags are not transitive in the install environment; we
# don't want to force downstreams to, e.g., compile with "-Werror"
# just because we wanted to compile that way.
target_link_libraries(H2Core PRIVATE
  $<BUILD_INTERFACE:$<TARGET_NAME_IF_EXISTS:h2_coverage_flags>>)
target_link_libraries(H2Core PUBLIC
  spdlog::spdlog
  ${H2_CUDA_LIBS}
  ${H2_ROCM_LIBS}
  $<$<BOOL:${H2_HAS_MPI}>:MPI::MPI_CXX>)

install(TARGETS H2Core
  EXPORT DiHydrogenTargets
  COMPONENT CXX)

list(APPEND H2_LIBRARIES H2Core)

# Setup the testing
include(CTest)
add_subdirectory(test)

# Setup clang format
if (TARGET clang-format)
  add_clang_format_to_all_targets(TARGETS H2Core H2Meta)
endif ()

# Setup the documentation targets
add_subdirectory(doc)

# The legacy distconv code
if (H2_ENABLE_DISTCONV_LEGACY)
  add_subdirectory(legacy)
endif ()

# Install target stuff
include (CMakePackageConfigHelpers)

# The version file is the same for install and build directories.
write_basic_package_version_file(
  "${CMAKE_BINARY_DIR}/DiHydrogenConfigVersion.cmake"
  COMPATIBILITY SameMinorVersion)

# Build tree export
export(EXPORT DiHydrogenTargets
  NAMESPACE H2::
  FILE DiHydrogenTargets.cmake)

# Write the configure file for the build tree
set(INCLUDE_INSTALL_DIRS
  "${CMAKE_SOURCE_DIR}/include/h2"
  "${CMAKE_BINARY_DIR}" # h2_config.hpp lives here.
  "${CMAKE_BINARY_DIR}/include/h2")
set(LIB_INSTALL_DIR "${CMAKE_BINARY_DIR}")
set(EXTRA_CMAKE_MODULE_DIR "${CMAKE_MODULE_DIR}")
configure_package_config_file(
  "${CONFIG_FILE_DIR}/DiHydrogenConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/DiHydrogenConfig.cmake"
  INSTALL_DESTINATION "${CMAKE_BINARY_DIR}"
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Now the install tree...
set(INCLUDE_INSTALL_DIRS ${CMAKE_INSTALL_INCLUDEDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(CMAKE_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR}/cmake/dihydrogen)
unset(EXTRA_CMAKE_MODULE_DIR)

# Install tree export
install(EXPORT DiHydrogenTargets
  NAMESPACE H2::
  DESTINATION ${CMAKE_INSTALL_DIR})

# Write the configure file for the install tree
configure_package_config_file(
  "${CONFIG_FILE_DIR}/DiHydrogenConfig.cmake.in"
  "${CMAKE_BINARY_DIR}/DiHydrogenConfig.install.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_DIR}
  PATH_VARS INCLUDE_INSTALL_DIRS LIB_INSTALL_DIR)

# Install the CMake stuff
install(FILES
  "${PROJECT_BINARY_DIR}/DiHydrogenConfig.install.cmake"
  RENAME "DiHydrogenConfig.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(FILES
  "${PROJECT_BINARY_DIR}/DiHydrogenConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_DIR})
install(DIRECTORY ${CMAKE_MODULE_DIR}
  DESTINATION ${CMAKE_INSTALL_DIR}
  FILES_MATCHING PATTERN "*.cmake")
