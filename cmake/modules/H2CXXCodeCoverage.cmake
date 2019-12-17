# There's two branches here; LLVM is obviously preferred.
#
# This will produce a macro with the following signature:
#
#   macro(add_code_coverage EXE_TARGET MASTER_TARGET)
#
#  Where EXE_TARGET is the target to which to add coverage and
#  MASTER_TARGET is a "dummy target" that depends on all of the
#  individual coverage targets. This will make it possible to run,
#  e.g., "ninja coverage" to generate all of the coverage targets.
#
# TODO: Expand add_code_coverage to handle libraries
# TODO: Have add_code_coverage add "linkage" to the compiler flags
# TODO: Expand add_code_coverage to build one coverage report for all
#       covered targets
# TODO: Better documentation of add_code_coverage macro
# TODO: Should add_code_coverage be a function?? It works now as a
#       macro, but this might be worth considering.
# TODO: Maybe clean up redirection in executing commands. Probably not
#       platform-independent...
#         --> This can be done by writing CMake scripts that wrap all
#             calls in `execute_process` abd setting the appropriate
#             OUTPUT_* and ERROR_* files.

# Verify we have an ok compiler
if (NOT ((CMAKE_CXX_COMPILER_ID MATCHES GNU) OR
      (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")))
  message(FATAL_ERROR
    "Coverage tools only supported for GCC and Clang compilers.\n"
    "Compiler detected as: ${CMAKE_CXX_COMPILER_ID}\n"
    "If this compiler supports coverage, please open an issue.")
endif ()

#
# Look for coverage tools.
#

# Coverage tools are often bundled with the compiler; this might help
# search for things.
get_filename_component(COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(COMPILER_PREFIX "${COMPILER_BIN_DIR}" DIRECTORY)

# Use the (far superior) LLVM tools if using the (far superior) LLVM
# compilers (or the Apple versions of them).
#
# If, for whatever reason, these are not found, the LLVM compilers can
# still use the GCOV tools.
if (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
  find_program(LLVM_COV_PROGRAM llvm-cov
    HINTS ${COMPILER_BIN_DIR}
    DOC "The llvm-cov program."
    NO_DEFAULT_PATH)
  find_program(LLVM_COV_PROGRAM llvm-cov
    HINTS ${LLVM_DIR} $ENV{LLVM_DIR}
    PATH_SUFFIXES bin
    DOC "The llvm-cov program."
    NO_DEFAULT_PATH)
  find_program(LLVM_COV_PROGRAM llvm-cov)

  find_program(LLVM_PROFDATA_PROGRAM llvm-profdata
    HINTS ${COMPILER_BIN_DIR}
    DOC "The llvm-profdata program."
    NO_DEFAULT_PATH)
  find_program(LLVM_PROFDATA_PROGRAM llvm-profdata
    HINTS ${LLVM_DIR} $ENV{LLVM_DIR}
    PATH_SUFFIXES bin
    DOC "The llvm-profdata program."
    NO_DEFAULT_PATH)
  find_program(LLVM_PROFDATA_PROGRAM llvm-profdata)
endif ()

if (LLVM_COV_PROGRAM AND LLVM_PROFDATA_PROGRAM)
  set(H2_HAVE_LLVM_COVERAGE_TOOLS TRUE)

  message(STATUS "Found llvm-cov: ${LLVM_COV_PROGRAM}")
  message(STATUS "Found llvm-profdata: ${LLVM_PROFDATA_PROGRAM}")
  message(STATUS "Using LLVM coverage tools.")
endif ()

if (NOT H2_HAVE_LLVM_COVERAGE_TOOLS)
  if (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
    set(GCOV_EXE_NAME "llvm-cov")
  else ()
    set(GCOV_EXE_NAME "gcov")
  endif ()

  find_program(GCOV_PROGRAM ${GCOV_EXE_NAME}
    HINTS ${COMPILER_BIN_DIR}
    DOC "The gcov coverage tool."
    NO_DEFAULT_PATH)
  find_program(GCOV_PROGRAM ${GCOV_EXE_NAME}
    HINTS ${GCOV_DIR} $ENV{GCOV_DIR}
    DOC "The gcov coverage tool."
    NO_DEFAULT_PATH)
  find_program(GCOV_PROGRAM ${GCOV_EXE_NAME})

  find_program(LCOV_PROGRAM lcov
    HINTS ${LCOV_DIR} $ENV{LCOV_DIR}
    DOC "The lcov coverage tool."
    NO_DEFAULT_PATH)
  find_program(LCOV_PROGRAM lcov)

  find_program(GENHTML_PROGRAM genhtml
    HINTS ${GENHTML_DIR} $ENV{GENHTML_DIR}
    DOC "The genhtml tool."
    NO_DEFAULT_PATH)
  find_program(GENHTML_PROGRAM genhtml)

  if (GCOV_PROGRAM AND LCOV_PROGRAM AND GENHTML_PROGRAM)
    set(H2_HAVE_GCOV_COVERAGE_TOOLS ON)
    message(STATUS "Found gcov: ${GCOV_PROGRAM}")
    message(STATUS "Found lcov: ${LCOV_PROGRAM}")
    message(STATUS "Found genhtml: ${GENHTML_PROGRAM}")
    message(STATUS "Using GCOV coverage tools.")

    if (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang")
      # Admittedly, this is a bit pathological. This would imply that
      # "llvm-cov" is found but "llvm-profdata" isn't found.
      file(WRITE "${CMAKE_BINARY_DIR}/coverage/tmp/llvm-gcov.sh"
        "#! /bin/bash
exec ${GCOV_PROGRAM} gcov $@")
      file(COPY "${CMAKE_BINARY_DIR}/coverage/tmp/llvm-gcov.sh"
        DESTINATION "${CMAKE_BINARY_DIR}/coverage/"
        FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
      file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/coverage/tmp")
      set(GCOV_PROGRAM "${CMAKE_BINARY_DIR}/coverage/llvm-gcov.sh")
    endif ()
  endif ()
endif ()

if (NOT (H2_HAVE_LLVM_COVERAGE_TOOLS OR H2_HAVE_GCOV_COVERAGE_TOOLS))
  message(FATAL_ERROR
    "No suitable coverage tools were found."
    "Please install the LLVM coverage tools, llvm-cov and llvm-profdata, or\n"
    "the GCOV coverage tools, gcov, lcov, and genhtml.")
endif ()

set(LLVM_COVERAGE_FLAGS
  "-fprofile-instr-generate" "-fcoverage-mapping")

set(GCOV_COVERAGE_FLAGS
   "--coverage" "-fprofile-arcs" "-ftest-coverage")

# We want to use the LLVM coverage flags if compiler is clang and LLVM
# coverage tools.
unset(COVERAGE_FLAGS)
if (H2_HAVE_LLVM_COVERAGE_TOOLS AND
    (CMAKE_CXX_COMPILER_ID MATCHES "(Apple)?[Cc]lang"))
  set(COVERAGE_FLAGS ${LLVM_COVERAGE_FLAGS})

  macro(add_code_coverage EXE_TARGET MASTER_TARGET)
    set(_PROF_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${EXE_TARGET}/prof")
    set(_HTML_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${EXE_TARGET}/html")

    add_custom_target(
      ${EXE_TARGET}-gen-profdata
      COMMAND LLVM_PROFILE_FILE=${_PROF_OUT_DIR}/${EXE_TARGET}.profraw $<TARGET_FILE:${EXE_TARGET}> &> /dev/null
      COMMAND ${LLVM_PROFDATA_PROGRAM} merge -sparse ${_PROF_OUT_DIR}/${EXE_TARGET}.profraw -o ${_PROF_OUT_DIR}/${EXE_TARGET}.profdata
      BYPRODUCTS ${_PROF_OUT_DIR}/${EXE_TARGET}.profdata
      DEPENDS ${EXE_TARGET}
      COMMENT "Generating coverage data for ${EXE_TARGET}."
      VERBATIM)

    add_custom_target(
      ${EXE_TARGET}-gen-coverage-html
      COMMAND ${LLVM_COV_PROGRAM} show $<TARGET_FILE:${EXE_TARGET}> -instr-profile=${_PROF_OUT_DIR}/${EXE_TARGET}.profdata -show-line-counts-or-regions -output-dir=${_HTML_OUT_DIR} -format=html
      BYPRODUCTS ${_HTML_OUT_DIR}/index.html
      DEPENDS ${EXE_TARGET}-gen-profdata
      COMMENT "Generating HTML for ${EXE_TARGET} coverage report."
      VERBATIM)

    add_custom_target(
      ${EXE_TARGET}-coverage
      COMMAND ${CMAKE_COMMAND} -E echo "Open ${_HTML_OUT_DIR}/index.html in a web browser to view report."
      DEPENDS ${EXE_TARGET}-gen-coverage-html
      COMMENT "Generated coverage report for ${EXE_TARGET}.")

    add_dependencies(${MASTER_TARGET} ${EXE_TARGET}-coverage)
  endmacro()

  # clang++ -fprofile-instr-generate -fcoverage-mapping -O0 test.cpp
  # LLVM_PROFILE_FILE=test.profraw ./a.out
  # llvm-profdata merge -sparse test.profraw -o test.profdata
  #
  # Command line reports:
  #  llvm-cov show a.out -instr-profile=test.profdata -show-line-counts-or-regions
  #  llvm-cov report a.out -instr-profile=test.profdata
  #
  # HTML report:
  #  llvm-cov show a.out -instr-profile=test.profdata -show-line-counts-or-regions -output-dir=test_html -format="html"
  #  open test_html/index.html

elseif (H2_HAVE_GCOV_COVERAGE_TOOLS)

  set(COVERAGE_FLAGS ${GCOV_COVERAGE_FLAGS})

  macro(add_code_coverage EXE_TARGET MASTER_TARGET)
    set(_INFO_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${EXE_TARGET}/info")
    set(_HTML_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${EXE_TARGET}/html")

    add_custom_target(
      ${EXE_TARGET}-gen-lcov
      # Reset lcov state
      COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --zerocounters
      # Fix files with no called functions
      COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --base-directory ${CMAKE_BINARY_DIR} --capture --initial --output-file ${_INFO_OUT_DIR}/${EXE_TARGET}.base
      # Run exe and dump coverage data
      COMMAND $<TARGET_FILE:${EXE_TARGET}> &> /dev/null
      # Gather coverage data
      COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --capture --directory ${CMAKE_BINARY_DIR} --output-file ${_INFO_OUT_DIR}/${EXE_TARGET}.info
      # Add everything together
      COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -a ${_INFO_OUT_DIR}/${EXE_TARGET}.base -a ${_INFO_OUT_DIR}/${EXE_TARGET}.info --output-file ${_INFO_OUT_DIR}/${EXE_TARGET}.total.info.tmp
      COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -r ${_INFO_OUT_DIR}/${EXE_TARGET}.total.info.tmp "*v1/*" "${COMPILER_PREFIX}/*" -o ${_INFO_OUT_DIR}/${EXE_TARGET}.total.info.final
      BYPRODUCTS ${_INFO_OUT_DIR}/${EXE_TARGET}.total.info.final
      DEPENDS ${EXE_TARGET}
      COMMENT "Generating coverage data for ${EXE_TARGET}."
      VERBATIM)

    add_custom_target(
      ${EXE_TARGET}-gen-coverage-html
      COMMAND ${GENHTML_PROGRAM} ${_INFO_OUT_DIR}/${EXE_TARGET}.total.info.final --output-directory ${_HTML_OUT_DIR}
      DEPENDS ${EXE_TARGET}-gen-lcov
      COMMENT "Generating HTML for ${EXE_TARGET} coverage report."
      BYPRODUCTS ${_HTML_OUT_DIR}/index.html
      VERBATIM)

    add_custom_target(
      ${EXE_TARGET}-coverage
      COMMAND ${CMAKE_COMMAND} -E echo "Open ${_HTML_OUT_DIR}/index.html in a web browser to view report."
      DEPENDS ${EXE_TARGET}-gen-coverage-html
      COMMENT "Generated coverage report for ${EXE_TARGET}.")

    add_dependencies(${MASTER_TARGET} ${EXE_TARGET}-coverage)

  endmacro()

  # g++ --coverage -ftest-coverage -fprofile-arcs file.cpp
  # ./a.out
  # lcov --capture --directory . --output-file file.info
  # genhtml file.info --output-directory out
endif ()

add_library(h2_coverage_flags INTERFACE)
target_compile_options(h2_coverage_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${COVERAGE_FLAGS}>)
target_link_options(h2_coverage_flags INTERFACE ${COVERAGE_FLAGS})

add_custom_target(clean-coverage
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage
  COMMENT "Cleaning up coverage data."
  VERBATIM)
