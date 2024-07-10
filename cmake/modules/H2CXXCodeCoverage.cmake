################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# There's two branches here; LLVM is obviously preferred.
#
# This will produce a macro with the following signature:
#
#   macro(add_code_coverage EXE_TARGET TOPLEVEL_TARGET)
#
#  Where EXE_TARGET is the target to which to add coverage and
#  TOPLEVEL_TARGET is a "dummy target" that depends on all of the
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
    "Compiler Path: ${CMAKE_CXX_COMPILER}\n"
    "Compiler ID: ${CMAKE_CXX_COMPILER_ID}\n"
    "If this compiler supports coverage, please open an issue.")
endif ()

# Look for coverage tools.
#
# If we are doing a CI build, we need to prefer GCOV-compatible tools
# (gcovr->cobertura format). Otherwise, prefer the toolchain
# associated with the compiler.
#
# For GCOV tools, we should look for "gcov" if using GCC or "llvm-cov"
# if using an LLVM-based compiler. For LLVM-based coverage, we need
# "llvm-profdata".
#
# gcc workflow with GCC:
#   > g++ -O0 --coverage foo.cpp -o foo
#   > ./foo
#   > gcov -m -r -s ${PWD} foo.cpp
#
# gcc workflow with llvm:
#   > clang++ -O0 --coverage foo.cpp -o foo
#   > ./foo
#   > llvm-cov gcov -m -r -s ${PWD} foo.cpp
#
# llvm workflow:
#   > clang++ -O0 -fprofile-instr-generate -fcoverage-mapping foo.cpp -o foo
#   > LLVM_PROFILE_FILE=foo.profraw ./foo
#   > llvm-profdata merge -sparse foo.profraw -o foo.profdata
#   > llvm-cov report ./foo --instr-profile=foo.profdata
#
# To generate HTML reports, GCC tools will prefer "lcov"+"genhtml",
# whereas LLVM-based tooling will use "llvm-cov show".
#
# In order to simplify the logic here, we only support GCC+GCC
# tooling, LLVM+LLVM-GCC tooling, LLVM+LLVM tooling.

# Coverage tools are often bundled with the compiler; this might help
# search for things.
get_filename_component(COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(COMPILER_PREFIX "${COMPILER_BIN_DIR}" DIRECTORY)

# On LC systems, using the "-magic" compilers might mean the compiler
# is actually a wrapper that is located separately from its associated
# tooling. Fortunately, the "real" compilers (and their tools) are at
# the same prefix with all the "-magic" stripped out...
#
# A common test for LC-ness is checking for the "SYS_TYPE" environment
# variable, so let's do that and call it good.
if (DEFINED ENV{SYS_TYPE})
  string(REPLACE "-magic" "" COMPILER_BIN_DIR "${COMPILER_BIN_DIR}")
endif ()

# On AMD systems with ROCm, we probably want the "llvm-amdgpu" tools,
# which we encode in the HIP compiler. So let's search that, too.
if (CMAKE_HIP_COMPILER AND (CMAKE_HIP_COMPILER_ID MATCHES "[Cc]lang"))
  get_filename_component(EXTRA_COMPILER_BIN_DIR
    "${CMAKE_HIP_COMPILER}"
    DIRECTORY)
  get_filename_component(EXTRA_COMPILER_PREFIX
    "${EXTRA_COMPILER_BIN_DIR}"
    DIRECTORY)
else ()
  set(EXTRA_COMPILER_BIN_DIR "${COMPILER_BIN_DIR}")
  set(EXTRA_COMPILER_PREFIX "${COMPILER_PREFIX}")
endif ()

# First, look for the appropriate coverage tools
if (CMAKE_CXX_COMPILER_ID MATCHES GNU)
  # First search with the compiler...
  find_program(GCOV_PROGRAM gcov
    HINTS ${COMPILER_BIN_DIR}
    DOC "The gcov coverage tool."
    NO_DEFAULT_PATH)
  # Then default paths.
  find_program(GCOV_PROGRAM gcov)

elseif (CMAKE_CXX_COMPILER_ID MATCHES "[Cc]lang")

  find_program(LLVM_PROFDATA_PROGRAM llvm-profdata
    HINTS ${COMPILER_BIN_DIR} ${EXTRA_COMPILER_BIN_DIR}
    DOC "The llvm-profdata program."
    NO_DEFAULT_PATH)
  find_program(LLVM_PROFDATA_PROGRAM llvm-profdata)

  find_program(LLVM_COV_PROGRAM llvm-cov
    HINTS ${COMPILER_BIN_DIR} ${EXTRA_COMPILER_BIN_DIR}
    DOC "The llvm-cov program."
    NO_DEFAULT_PATH)
  find_program(LLVM_COV_PROGRAM llvm-cov)

  if (H2_CI_BUILD)

    # NOTE: We need a single exe to pass to lcov. So we make one.
    file(WRITE "${CMAKE_BINARY_DIR}/coverage/tmp/llvm-gcov.sh"
      "#! /bin/bash
${LLVM_COV_PROGRAM} gcov -m $@")
    file(COPY "${CMAKE_BINARY_DIR}/coverage/tmp/llvm-gcov.sh"
      DESTINATION "${CMAKE_BINARY_DIR}"
      FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE)
    file(REMOVE_RECURSE "${CMAKE_BINARY_DIR}/coverage/tmp")
    set(GCOV_PROGRAM "${CMAKE_BINARY_DIR}/llvm-gcov.sh")

  endif (H2_CI_BUILD)
endif ()

if (GCOV_PROGRAM)
  find_program(LCOV_PROGRAM lcov)

  if (LCOV_PROGRAM)
    get_filename_component(LCOV_BIN_DIR "${LCOV_PROGRAM}" DIRECTORY)
    find_program(GENHTML_PROGRAM genhtml
      HINTS ${LCOV_BIN_DIR}
      NO_DEFAULT_PATH)
  endif ()
  find_program(GENHTML_PROGRAM genhtml)

  if (H2_CI_BUILD)
    find_program(GCOVR_PROGRAM gcovr)
    if (NOT GCOVR_PROGRAM)
      message(STATUS "Could not find gcovr.")
    endif ()
  endif ()

  if (GCOV_PROGRAM AND LCOV_PROGRAM AND GENHTML_PROGRAM)
    set(H2_HAVE_GCOV_COVERAGE_TOOLS ON)
    set(H2_COVERAGE_FLAGS
      "--coverage" "-fprofile-arcs" "-ftest-coverage" "-O0")
    message(STATUS "Found gcov: ${GCOV_PROGRAM}")
    message(STATUS "Found lcov: ${LCOV_PROGRAM}")
    message(STATUS "Found genhtml: ${GENHTML_PROGRAM}")
    if (GCOVR_PROGRAM)
      message(STATUS "Found gcovr: ${GCOVR_PROGRAM}")
    endif ()
    message(STATUS "Using GCOV coverage tools.")
  endif ()
endif ()

# In this case, you are using LLVM toolchains and EITHER
# H2_CI_BUILD=OFF OR you have failed to find lcov/genhtml. In the
# latter case, We can still generate HTML reports, they just won't
# integrate with the GitLab CI stuff (gcovr).
if (LLVM_COV_PROGRAM AND LLVM_PROFDATA_PROGRAM
    AND NOT H2_HAVE_GCOV_COVERAGE_TOOLS)
  set(H2_HAVE_LLVM_COVERAGE_TOOLS TRUE)
  set(H2_COVERAGE_FLAGS
    "-fprofile-instr-generate" "-fcoverage-mapping" "-O0")
  message(STATUS "Found llvm-cov: ${LLVM_COV_PROGRAM}")
  message(STATUS "Found llvm-profdata: ${LLVM_PROFDATA_PROGRAM}")
  message(STATUS "Using LLVM coverage tools.")
endif ()

if (NOT H2_HAVE_LLVM_COVERAGE_TOOLS AND NOT H2_HAVE_GCOV_COVERAGE_TOOLS)
  message(FATAL_ERROR
    "No suitable coverage tools were found.\n"
    "For GCC-based tools, it may be useful to set:\n"
    "  GCOV_ROOT, LCOV_ROOT, GENHTML_ROOT\n"
    "For LLVM-based tools, it may be useful to set:\n"
    "  LLVM-COV_ROOT, LLVM-PROFDATA_ROOT")
endif ()

define_property(GLOBAL PROPERTY COVERAGE_TARGETS
  BRIEF_DOCS
  "A list of all targets with coverage added to them")

define_property(TARGET PROPERTY COVERAGE_TARGETS
  BRIEF_DOCS
  "A list of the toplevel (dummy) coverage targets for this target")

if (H2_HAVE_GCOV_COVERAGE_TOOLS)
  set(COVERAGE_FLAGS ${GCOV_COVERAGE_FLAGS})

  macro(add_code_coverage EXE_TARGET TOPLEVEL_TARGET)
    # Add this target to the global list
    get_property(_all_cov_tgts GLOBAL PROPERTY COVERAGE_TARGETS)
    if (_all_cov_tgts)
      list(APPEND _all_cov_tgts "${EXE_TARGET}")
    else ()
      set(_all_cov_tgts "${EXE_TARGET}")
    endif ()
    set_property(GLOBAL PROPERTY COVERAGE_TARGETS "${_all_cov_tgts}")

    get_target_property(_tgt_coverage_tgts ${EXE_TARGET} COVERAGE_TARGETS)
    if (_tgt_coverage_tgts)
      list(APPEND _tgt_coverage_tgts "${TOPLEVEL_TARGET}")
    else ()
      set(_tgt_coverage_tgts "${TOPLEVEL_TARGET}")
    endif ()
    set_target_properties(${EXE_TARGET}
      PROPERTIES COVERAGE_TARGETS "${_tgt_coverage_tgts}")
  endmacro ()

  macro(finalize_code_coverage)

    # Find all targets
    get_property(_all_tgts GLOBAL PROPERTY COVERAGE_TARGETS)
    list(REMOVE_DUPLICATES _all_tgts)

    # Find all toplevel targets
    set(_all_toplevel_tgts)
    foreach (_tgt IN LISTS _all_tgts)
      get_target_property(_tgt_toplevel_tgts ${_tgt} COVERAGE_TARGETS)
      if (_tgt_toplevel_tgts)
        foreach (_tltgt IN LISTS _tgt_toplevel_tgts)
          list(APPEND _cov_tgts_${_tltgt} "${_tgt}")
          list(APPEND _all_toplevel_tgts "${_tltgt}")
        endforeach ()
      endif ()
    endforeach ()
    list(REMOVE_DUPLICATES _all_toplevel_tgts)

    # Build out each toplevel target
    foreach (_tltgt IN LISTS _all_toplevel_tgts)
      set(_INFO_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${_tltgt}/info")
      set(_HTML_OUT_DIR "${CMAKE_BINARY_DIR}/coverage/${_tltgt}/html")

      # Write a quick script to run all the executables. Note that
      # this only works for non-MPI executables.
      #
      # FIXME (trb 2024/07/10): This needs to work for MPI executables
      #                         as well.
      set(_run_all_tgts_src "#!/bin/bash\n")
      foreach (_tgt IN LISTS _cov_tgts_${_tltgt})
        string(APPEND _run_all_tgts_src "$<TARGET_FILE:${_tgt}> &> /dev/null\n")
      endforeach ()
      file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/coverage_${_tltgt}_run_all_tgts.sh"
        CONTENT "${_run_all_tgts_src}"
        FILE_PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
      )

      # Build a target
      add_custom_target(
        ${_tltgt}-gen-lcov
        # Reset lcov state
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --zerocounters
        # Fix files with no called functions
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --directory ${CMAKE_BINARY_DIR} --capture --initial --output-file ${_INFO_OUT_DIR}/${_tltgt}.base
        # Run exe and dump coverage data
        COMMAND "${CMAKE_BINARY_DIR}/coverage_${_tltgt}_run_all_tgts.sh"
        # Gather coverage data
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} --capture --directory ${CMAKE_BINARY_DIR} --output-file ${_INFO_OUT_DIR}/${_tltgt}.info
        # Add everything together
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -a ${_INFO_OUT_DIR}/${_tltgt}.base -a ${_INFO_OUT_DIR}/${_tltgt}.info --output-file ${_INFO_OUT_DIR}/${_tltgt}.total.info.tmp
        # Remove compiler stuff
        # COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -r ${_INFO_OUT_DIR}/${_tltgt}.total.info.tmp "*v1/*" "${COMPILER_PREFIX}/*" "${EXTRA_COMPILER_PREFIX}/*" -o ${_INFO_OUT_DIR}/${_tltgt}.total.info.final
        # Extract this project stuff
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -e ${_INFO_OUT_DIR}/${_tltgt}.total.info.tmp "${CMAKE_SOURCE_DIR}/*" "${CMAKE_BINARY_DIR}/*" -o ${_INFO_OUT_DIR}/${_tltgt}.total.info.almost.final
        # Remove CI stuff
        COMMAND ${LCOV_PROGRAM} --gcov-tool ${GCOV_PROGRAM} -r ${_INFO_OUT_DIR}/${_tltgt}.total.info.almost.final "${CMAKE_SOURCE_DIR}/install-deps*" "${CMAKE_SOURCE_DIR}/test/*" -o ${_INFO_OUT_DIR}/${_tltgt}.total.info.final
        BYPRODUCTS ${_INFO_OUT_DIR}/${_tltgt}.total.info.final
        COMMENT "Generating coverage data for \"${_tltgt}\"."
        VERBATIM)

      # Generate the HTML reports
      add_custom_target(
        ${_tltgt}-gen-coverage-html
        COMMAND ${GENHTML_PROGRAM} ${_INFO_OUT_DIR}/${_tltgt}.total.info.final --demangle-cpp --output-directory ${_HTML_OUT_DIR}
        COMMENT "Generating HTML for ${_tltgt} coverage report."
        BYPRODUCTS ${_HTML_OUT_DIR}/index.html
        VERBATIM)

      # Dummy target to just print out a nice message to the build log
      add_custom_target(
        ${_tltgt}-coverage
        COMMAND ${CMAKE_COMMAND} -E echo "Open ${_HTML_OUT_DIR}/index.html in a web browser to view report."
        COMMENT "Generated coverage report for target \"${_tltgt}\".")

      # Setup the dependency graph appropriately

      # The lcov target depends on each exe target being built. Make that explicit.
      foreach (tgt IN LISTS _cov_tgts_${_tltgt})
        add_dependencies(${_tltgt}-gen-lcov ${tgt})
      endforeach ()

      # Build each lcov target sequentially
      foreach (tgt IN LISTS _all_gen_lcov_tgts)
        add_dependencies(${_tltgt}-gen-lcov ${tgt})
      endforeach ()
      list(APPEND _all_gen_lcov_tgts ${_tltgt}-gen-lcov)

      # The HTML depends on the lcov
      add_dependencies(${_tltgt}-gen-coverage-html ${_tltgt}-gen-lcov)

      # The dummy message target depends on the HTML
      add_dependencies(${_tltgt}-coverage ${_tltgt}-gen-coverage-html)

      # The toplevel target depends on the dummy message target
      add_dependencies(${_tltgt} ${_tltgt}-coverage)

      if (GCOVR_PROGRAM)
        add_custom_target(
          ${_tltgt}-gcovr
          COMMAND ${GCOVR_PROGRAM} --cobertura-pretty --exclude-unreachable-branches --print-summary -o ${CMAKE_BINARY_DIR}/${_tltgt}-gcovr.xml --gcov-executable "${GCOV_PROGRAM} -b -c -f -r -s ${CMAKE_SOURCE_DIR}"
          COMMENT "Generating Cobertura report for \"${_tltgt}\" with gcovr."
          BYPRODUCTS ${CMAKE_BINARY_DIR}/${_tltgt}-gcovr.xml
          VERBATIM)
        # This comes *after* gen-coverage-html
        add_dependencies(${_tltgt}-gcovr ${_tltgt}-gen-coverage-html)
        # But is still driven by the coverage target
        add_dependencies(${_tltgt}-coverage ${_tltgt}-gcovr)
      endif ()
    endforeach ()
  endmacro()

  # g++ --coverage -ftest-coverage -fprofile-arcs file.cpp
  # ./a.out
  # lcov --capture --directory . --output-file file.info
  # genhtml file.info --output-directory out

elseif (H2_HAVE_LLVM_COVERAGE_TOOLS)

  macro(add_code_coverage EXE_TARGET TOPLEVEL_TARGET)
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

    add_dependencies(${TOPLEVEL_TARGET} ${EXE_TARGET}-coverage)
  endmacro()

  macro(finalize_code_coverage)
    # No-op
  endmacro ()

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
endif ()

add_library(h2_coverage_flags INTERFACE)
target_compile_options(h2_coverage_flags INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:${H2_COVERAGE_FLAGS}>
  $<$<COMPILE_LANGUAGE:HIP>:${H2_COVERAGE_FLAGS}>)
target_link_options(h2_coverage_flags INTERFACE ${H2_COVERAGE_FLAGS})

add_custom_target(clean-coverage
  COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/coverage
  COMMENT "Cleaning up coverage data."
  VERBATIM)
