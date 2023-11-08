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
