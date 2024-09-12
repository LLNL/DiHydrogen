################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# We'll have one of these for every coverage target specified in the
# project. Users will be required to specify the following variables:
#
# Project info:
#   COVERAGE_TGT: The name of the coverage target being built
#   SOURCE_DIR: The path to the toplevel project source code (probably
#               CMAKE_SOURCE_DIR)
#   BUILD_DIR: The path to the toplevel project build directory
#              (probably CMAKE_BINARY_DIR)
#   OUTPUT_DIR: The path to which output should be written
#   MPI_LAUNCH_PATTERN: The pattern to use when launching MPI. The
#                       string "<EXE>" will be replaced with the
#                       executable name.
#
# Coverage tools:
#   LCOV_PROGRAM: The lcov driver
#   GCOV_PROGRAM: The gcov executable or a suitable wrapper
#
# Coverage targets:
#   SEQ_COVERAGE_PROGRAMS: A list of programs that run sequentially
#                          (threaded ok, but no MPI).
#   MPI_COVERAGE_PROGRAMS: A list of programs to run with MPI

# TODO: Input checking

# Match H2's minimum required
cmake_minimum_required(VERSION 3.21.0)

message("----------------------------------------")
message("COVERAGE SCRIPT VARIABLES")
message("  COVERAGE_TGT: ${COVERAGE_TGT}")
message("  SOURCE_DIR: ${SOURCE_DIR}")
message("  BUILD_DIR: ${BUILD_DIR}")
message("  OUTPUT_DIR: ${OUTPUT_DIR}")
message("  MPI_LAUNCH_PATTERN: ${MPI_LAUNCH_PATTERN}")
message("")
message("  LCOV_PROGRAM: ${LCOV_PROGRAM}")
message("  GCOV_PROGRAM: ${GCOV_PROGRAM}")
message("")
message("  SEQ_COVERAGE_PROGRAMS: ${SEQ_COVERAGE_PROGRAMS}")
message("  MPI_COVERAGE_PROGRAMS: ${MPI_COVERAGE_PROGRAMS}")
message("----------------------------------------")

set(CMAKE_EXECUTE_PROCESS_COMMAND_ECHO "STDOUT")
set(OUTPUT_INFO_DIR "${OUTPUT_DIR}/info")
set(OUTPUT_TMP_DIR "${OUTPUT_DIR}/tmp")

# Execute the given process. 2 minute timeout to protect against hangs
# in the test drivers.
macro(h2_run_process)
  set(_command ${ARGN})
  execute_process(
    COMMAND ${_command}
    TIMEOUT 120
    RESULT_VARIABLE _result
    OUTPUT_VARIABLE _outvar
    ERROR_VARIABLE _errvar)
  if (_result)
    list(JOIN _command " " _cmd)
    set(_errmsg
      "\nError in command: ${_cmd}\nstdout: ${_outvar}\nstderr: ${_errvar}")
    message(FATAL_ERROR "${_errmsg}")
  endif ()
endmacro ()

# Reset lcov state
h2_run_process(
  "${LCOV_PROGRAM}"
  --gcov-tool "${GCOV_PROGRAM}"
  --directory "${BUILD_DIR}"
  --zerocounters)

# Capture initial output
h2_run_process(
  ${LCOV_PROGRAM}
  --gcov-tool ${GCOV_PROGRAM}
  --directory ${BUILD_DIR}
  --capture --initial
  --output-file ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info)

foreach (seq_exe IN LISTS SEQ_COVERAGE_PROGRAMS)
  # Run exe and dump coverage data
  h2_run_process(${seq_exe})
endforeach ()

# Gather all sequential coverage data
h2_run_process(
  ${LCOV_PROGRAM}
  --gcov-tool ${GCOV_PROGRAM}
  --capture
  --directory ${BUILD_DIR}
  --output-file ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info)

foreach (mpi_exe IN LISTS MPI_COVERAGE_PROGRAMS)
  string(REPLACE "<EXE>" "${mpi_exe}" _mpi_command_tmp "${MPI_LAUNCH_PATTERN}")
  string(REPLACE "\"" "" _mpi_command_tmp "${_mpi_command_tmp}")
  string(REPLACE " " ";" _mpi_command "${_mpi_command_tmp}")

  # Run the binary
  h2_run_process(${_mpi_command})

  # Copy each rank's data into the build tree so it can be captured
  # correctly. Then capture it.
  file(GLOB _gcov_prefixes "${OUTPUT_TMP_DIR}/gcov-*")
  foreach (_prefix IN LISTS _gcov_prefixes)

    # Use this over "file(COPY ...)" since that creates a copy of the
    # source directory within the destination directory, whereas
    # `copy_directory` copies the *contents* of the
    # source directory within the destination.
    h2_run_process(
      ${CMAKE_COMMAND} -E
      copy_directory "${_prefix}" "${BUILD_DIR}")

    # Remove stuff so gcovr/etc doesn't screw things up.
    h2_run_process(
      ${CMAKE_COMMAND} -E rm -r "${_prefix}")

    # Update coverage counts
    h2_run_process(
      ${LCOV_PROGRAM}
      --gcov-tool ${GCOV_PROGRAM}
      --capture
      --directory ${BUILD_DIR}
      --output-file ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info)
  endforeach ()
endforeach ()

# Extract this project stuff
h2_run_process(
  ${LCOV_PROGRAM}
  --gcov-tool ${GCOV_PROGRAM}
  -e ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info
  "${SOURCE_DIR}/*" "${BUILD_DIR}/*"
  -o ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info)

# Remove CI stuff
h2_run_process(
  ${LCOV_PROGRAM}
  --gcov-tool ${GCOV_PROGRAM}
  -r ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.info
  "${SOURCE_DIR}/install-deps*" "${SOURCE_DIR}/test/*"
  -o ${OUTPUT_INFO_DIR}/${COVERAGE_TGT}.final.info)
