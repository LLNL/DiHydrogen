################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

include(CheckCXXSourceCompiles)
include(CheckCXXCompilerFlag)

# Determine if the C++ compiler has __PRETTY_FUNCTION__. Otherwise it
# returns the standardized __FUNC__.
function (h2_cxx_determine_pretty_function OUTVAR)
  set(h2_detect_pretty_function_src__
    "int main()
{
  char const* str = __PRETTY_FUNCTION__;
  (void) str;
  return 0;
}")

  check_cxx_source_compiles(
    "${h2_detect_pretty_function_src__}"
    H2_CXX_HAS_PRETTY_FUNCTION)

  if (H2_CXX_HAS_PRETTY_FUNCTION)
    set(${OUTVAR} "__PRETTY_FUNCTION__" PARENT_SCOPE)
  else ()
    set(${OUTVAR} "__FUNC__" PARENT_SCOPE)
  endif ()
endfunction ()

# Determine whether the C++ compiler supports a version of the
# "restrict" compiler. While this is a standardized feature in C, it's
# not standard C++.
function (h2_cxx_determine_restrict_qualifier OUTVAR)
  set(restrict_candidates__
    "__restrict__"
    "__restrict"
    "restrict"
    "__declspec(restrict)")

  foreach (qual ${restrict_candidates__})
    string(TOUPPER ${qual} QUAL_UPPER_TMP)
    string(REGEX REPLACE "[)(]" "_" QUAL_UPPER ${QUAL_UPPER_TMP})
    set(h2_detect_restrict_src__
      "void f(int * ${qual} a) { (void)a; } int main() {return 0;}")
    check_cxx_source_compiles(
      "${h2_detect_restrict_src__}"
      H2_CXX_HAS_${QUAL_UPPER})
    if (H2_CXX_HAS_${QUAL_UPPER})
      set(${OUTVAR} ${qual} PARENT_SCOPE)
      return ()
    endif ()
  endforeach ()
  unset(${OUTVAR} PARENT_SCOPE)
endfunction ()

# Attempt to setup warning flags for developer mode. Appends to the
# variable passed in OUTVAR. This is very GNU/Clang-centric.
function (h2_cxx_get_developer_warning_flags OUTVAR)
  set(_FOUND_CXX_FLAGS)
  list(APPEND _CANDIDATE_CXX_FLAGS
    "-Wall" "-Wextra" "-Wpedantic" "-pedantic")

  foreach (flag IN LISTS _CANDIDATE_CXX_FLAGS)
    string(REGEX REPLACE "^-" "" _flag_no_dash "${flag}")
    string(TOUPPER "${_flag_no_dash}" _flag_upper)
    check_cxx_compiler_flag("${flag}" CXX_HAS_FLAG_${_flag_upper})
    if (CXX_HAS_FLAG_${_flag_upper})
      list(APPEND _FOUND_CXX_FLAGS "${flag}")
    endif ()
  endforeach ()
  set(${OUTVAR} ${${OUTVAR}} ${_FOUND_CXX_FLAGS} PARENT_SCOPE)
endfunction ()

# Attempt to setup warnings-as-errors ("-Werror") flag. Currently only
# checks for "-Werror". If found, the flag is appended to the initial
# value of OUTVAR; if not found, OUTVAR is not changed.
function (h2_cxx_get_warnings_as_errors_flag OUTVAR)
  check_cxx_compiler_flag("-Werror" CXX_HAS_FLAG_WERROR)
  if (CXX_HAS_FLAG_WERROR)
    set(${OUTVAR} ${${OUTVAR}} "-Werror" PARENT_SCOPE)
  endif ()
endfunction ()
