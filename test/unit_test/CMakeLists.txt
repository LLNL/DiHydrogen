################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# TODO: GPU catch main(?)

# Setup this directory tests. It's not clear that all tests will
# always live here or if they'll scatter throughout the source, so
# making this possibly overly general (because the generality is so
# low-overhead).
h2_set_full_path(THIS_DIR_H2_SEQUENTIAL_CATCH_TESTS
  # None yet
  )

list(APPEND H2_SEQUENTIAL_CATCH_TESTS ${THIS_DIR_H2_SEQUENTIAL_CATCH_TESTS})

# Add a test for printing the version. This is mostly for getting
# CTest set up and verified.

add_executable(VersionTestExe VersionTest.cpp)
target_link_libraries(VersionTestExe PRIVATE ${H2_LIBRARIES})
set_target_properties(VersionTestExe
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

if (H2_EXTRA_CXX_FLAGS)
  target_compile_options(VersionTestExe
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${H2_EXTRA_CXX_FLAGS}>)
endif ()

add_test(NAME [==[Testing the Version String.]==]
  COMMAND $<TARGET_FILE:VersionTestExe>
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties([==[Testing the Version String.]==] PROPERTIES
  PASS_REGULAR_EXPRESSION "${PROJECT_VERSION}")

#
# Add the sequential Catch2 driver.
#

add_executable(SeqCatchTests)

# Add the MPI Catch2 driver.
add_executable(MPICatchTests MPICatchMain.cpp mpi_cumulative_reporter.cpp mpi_event_listener.cpp)
target_link_libraries(MPICatchTests
  PRIVATE ${H2_LIBRARIES} Catch2::Catch2)
set_target_properties(MPICatchTests
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

if (H2_HAS_GPU)
  add_executable(GPUCatchTests GPUCatchMain.cpp wait.cu)
  target_link_libraries(GPUCatchTests
    PRIVATE ${H2_LIBRARIES} Catch2::Catch2)
  target_compile_definitions(GPUCatchTests PRIVATE H2_TEST_WITH_GPU=1)
  set_target_properties(GPUCatchTests PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    CXX_STANDARD_REQUIRED ON)
  # MPI testing always tests with GPUs when built with them.
  target_compile_definitions(MPICatchTests PRIVATE H2_TEST_WITH_GPU=1)
  target_sources(MPICatchTests PRIVATE wait.cu)
  # Make CMake happy:
  if (H2_ENABLE_ROCM)
    set_source_files_properties(wait.cu PROPERTIES LANGUAGE HIP)
  endif ()
endif ()

# Add Catch2 unit tests
add_subdirectory(core)
add_subdirectory(patterns/factory)
add_subdirectory(patterns/multimethods)
add_subdirectory(utils)
add_subdirectory(tensor)

target_link_libraries(SeqCatchTests
  PRIVATE ${H2_LIBRARIES} Catch2::Catch2WithMain)
set_target_properties(SeqCatchTests
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

# Add the test. Outputs to console (for CTest's default logging) and
# in JUnit format for ease of CI integration
add_test(
  NAME "Sequential Catch2 Tests"
  COMMAND $<TARGET_FILE:SeqCatchTests> -r console -r JUnit::out=seq.xml
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

if (H2_CODE_COVERAGE)
  add_code_coverage(SeqCatchTests coverage)
  target_link_libraries(SeqCatchTests PRIVATE h2_coverage_flags)

  add_code_coverage(MPICatchTests coverage)
  target_link_libraries(MPICatchTests PRIVATE h2_coverage_flags)

  if (TARGET GPUCatchTests)
    add_code_coverage(GPUCatchTests coverage)
    target_link_libraries(GPUCatchTests PRIVATE h2_coverage_flags)
  endif ()
endif ()

if (H2_EXTRA_CXX_FLAGS)
  target_compile_options(SeqCatchTests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${H2_EXTRA_CXX_FLAGS}>)
endif ()
