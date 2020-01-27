# @H2_LICENSE_TEXT@

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
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

if (H2_EXTRA_CXX_FLAGS)
  target_compile_options(VersionTestExe
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${H2_EXTRA_CXX_FLAGS}>)
endif ()

if (H2_CODE_COVERAGE)
  add_code_coverage(VersionTestExe coverage)
  target_link_libraries(VersionTestExe PRIVATE h2_coverage_flags)
endif ()

add_test(NAME "[==[Testing the Version String.]==]"
  COMMAND $<TARGET_FILE:VersionTestExe>
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties("[==[Testing the Version String.]==]" PROPERTIES
  PASS_REGULAR_EXPRESSION "${PROJECT_VERSION}")

#
# Add the sequential Catch2 driver.
#

add_executable(SeqCatchTests SequentialCatchMain.cpp)

# Add Catch2 unit tests
add_subdirectory(patterns/factory)
add_subdirectory(patterns/multimethods)

target_link_libraries(SeqCatchTests
  PRIVATE ${H2_LIBRARIES} Catch2::Catch2)
set_target_properties(SeqCatchTests
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

if (H2_CODE_COVERAGE)
  add_code_coverage(SeqCatchTests coverage)
  target_link_libraries(SeqCatchTests PRIVATE h2_coverage_flags)
endif ()

if (H2_EXTRA_CXX_FLAGS)
  target_compile_options(SeqCatchTests
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${H2_EXTRA_CXX_FLAGS}>)
endif ()

if (H2_SEQUENTIAL_CATCH_TESTS)
  catch_discover_tests(SeqCatchTests TEST_PREFIX "Seq--")
endif ()
