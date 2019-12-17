# CUDA separate compilation is reported to work with v3.13.1, which is needed for NVSHMEM 
cmake_minimum_required(VERSION 3.9)

# Use the OLD policy for CMP0074. The NEW policy picks MPI_ROOT environment variable, which is automatically defined by the Spectrum-MPI module on Lassen. Apparently, it results in a compilation error due to -pthread being added to nvcc device link command. Another workaround is to unset MPI_ROOT when invoking CMake.
if ((${CMAKE_VERSION} VERSION_EQUAL 3.12.0) OR (${CMAKE_VERSION} VERSION_GREATER 3.12.0))
  cmake_policy(SET CMP0074 OLD)
endif ()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules)

project(DISTCONV CXX)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CMake build type" FORCE)
endif ()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  set(DISTCONV_DEBUG TRUE)
endif ()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(GNUInstallDirs)
include(SetupCXX)
include(SetupMPI)
include(SetupCUDAToolkit)
set(DISTCONV_HAS_CUDA ${CUDA_FOUND})
if (CUDA_FOUND)
  set(DISTCONV_CUDA_VERSION_MAJOR ${CUDA_VERSION_MAJOR})
  set(DISTCONV_CUDA_VERSION_MINOR ${CUDA_VERSION_MINOR})
endif ()

find_package(CUB REQUIRED)

# Options
option(DISTCONV_OPTIMIZE_FIND_DESTINATION
  "Enable optimization of find_destination." ON)

find_package(P2P)
set(DISTCONV_HAS_P2P ${P2P_FOUND})
# Use Aluminum's CMake config files
find_package(Aluminum NO_MODULE QUIET
  HINTS ${Aluminum_DIR} ${ALUMINUM_DIR} ${AL_DIR}
  $ENV{Aluminum_DIR} $ENV{ALUMINUM_DIR} $ENV{AL_DIR}
  PATH_SUFFIXES lib64/cmake/aluminum lib/cmake/aluminum
  NO_DEFAULT_PATH)
if (NOT Aluminum_FOUND)
  find_package(Aluminum NO_MODULE REQUIRED)
endif ()

# Use NVSHMEM if available
find_package(NVSHMEM)
set(DISTCONV_HAS_NVSHMEM ${NVSHMEM_FOUND})

find_package(cuDNN)
set(DISTCONV_HAS_CUDNN ${CUDNN_FOUND})

configure_file(
  "${CMAKE_SOURCE_DIR}/cmake/configure_files/distconv_config.hpp.in"
  "${CMAKE_BINARY_DIR}/distconv_config.hpp"
  @ONLY)

include(DISTCONVCMakeUtilities)
add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(benchmarks)

add_library(distconv ${DISTCONV_SOURCES} ${DISTCONV_HEADERS}
  ${DISTCONV_CUDA_SOURCES})

# PIC is required to link this into liblbann.so on x86
set_property(TARGET distconv PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NVSHMEM_FOUND)
  set_property(TARGET distconv PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()

target_include_directories(distconv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(distconv PUBLIC MPI::MPI_CXX)
target_link_libraries(distconv PUBLIC cuda::toolkit)
target_link_libraries(distconv PUBLIC cuda::CUB)
target_link_libraries(distconv PUBLIC cuda::cudnn)
if (P2P_FOUND)
  target_link_libraries(distconv PUBLIC P2P::P2P)
endif ()
target_link_libraries(distconv PUBLIC ${Aluminum_LIBRARIES})
if (NVSHMEM_FOUND)
  target_link_libraries(distconv PUBLIC NVSHMEM::NVSHMEM)
endif ()

install(
  TARGETS distconv
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/include/distconv"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  )

install(
  FILES "${CMAKE_BINARY_DIR}/distconv_config.hpp"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )