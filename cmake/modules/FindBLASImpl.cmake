# @H2_LICENSE_TEXT@

#[=============[.rst
FindBLASImpl
===============

Finds a reasonable BLAS implementation. This is basically a wrapper
around FindBLAS, with some extra functionality.

In addition to the variables set by FindBLAS, the following variables
will be defined::

  BLASImpl_FOUND   - True if the system has a compatible BLAS
                   implementation.
  BLAS_SUFFIX      - The mangling suffix for BLAS symbols.
  BLAS_IS_MKL      - True if the detected BLAS implementation is Intel MKL.
  BLAS_IS_OPENBLAS - True if the detected BLAS implementation is OpenBLAS.

The following cache variables will be set and marked as "advanced"::

  BLAS_MANGLING_USES_UNDERSCORE - True if an underscore is used for
                                  mangling BLAS symbols.
  BLAS_LIBRARY_IS_OPENBLAS      - True if OpenBLAS is detected.

In addition, the :prop_tgt:`IMPORTED` target ``BLAS::BLAS`` will
be created.

#]=============]
# This finds BLAS and does a few checks for additional functionality.
# In addition to the "normal" variables that FindBLAS sets, this will
# set the following variables:
#
#   BLAS_SUFFIX
#   BLAS_IS_MKL
#   BLAS_IS_OPENBLAS
#

include(CheckFunctionExists)
include(FindPackageHandleStandardArgs)

if (BLASImpl_FIND_REQUIRED)
  set(BLASImpl_REQUIRED_FLAG REQUIRED)
endif ()

if (BLASImpl_FIND_QUIETLY)
  set(BLASImpl_QUIET_FLAG QUIET)
endif ()

find_package(BLAS ${BLASImpl_QUIET_FLAG} ${BLASImpl_REQUIRED_FLAG})

if (BLAS_FOUND)
  if (BLASImpl_FIND_QUIETLY)
    set(CMAKE_REQUIRED_QUIET TRUE)
  endif ()

  set(CMAKE_REQUIRED_LIBRARIES "${BLAS_LINKER_FLAGS}" "${BLAS_LIBRARIES}")
  # Figure out the mangling
  check_function_exists(sgemm_ BLAS_MANGLING_USES_UNDERSCORE)
  if (BLAS_MANGLING_USES_UNDERSCORE)
    set(BLAS_SUFFIX "_")
  else ()
    set(BLAS_SUFFIX "")
  endif ()

  # Verify we have all the linkage we need
  set(REQUIRED_BLAS_FUNCTIONS
    saxpy scopy sdot snrm2 sgemv sgemm)

  set(BLAS_LIBRARY_HAS_REQUIRED TRUE)
  foreach (func IN LISTS REQUIRED_BLAS_FUNCTIONS)
    set(funcname "${func}${BLAS_SUFFIX}")
    string(TOUPPER "${func}" FUNCUPPER)
    check_function_exists(${funcname} BLAS_HAS_${FUNCUPPER})
    if (NOT BLAS_HAS_${FUNCUPPER})
      if (BLAS_LIBRARY_HAS_REQUIRED)
        set(BLAS_LIBRARY_HAS_REQUIRED FALSE)
      endif ()
      list(APPEND BLAS_MISSING_FUNCTIONS "${func}")
    endif ()
    mark_as_advanced(FORCE BLAS_HAS_${FUNCUPPER})
  endforeach ()

  set(MKL_FUNCTIONS mkl_simatcopy gemmt)
  set(BLAS_LIBRARY_IS_MKL TRUE)
  foreach (func IN LISTS MKL_FUNCTIONS)
    set(funcname "${func}${BLAS_SUFFIX}")
    string(TOUPPER "${func}" FUNCUPPER)
    check_function_exists(${funcname} BLAS_HAS_${FUNCUPPER})
    if (NOT BLAS_HAS_${FUNCUPPER} AND BLAS_LIBRARY_IS_MKL)
      set(BLAS_LIBRARY_IS_MKL FALSE)
      break ()
    endif ()
    mark_as_advanced(FORCE BLAS_HAS_${FUNCUPPER})
  endforeach ()

  # Check for OpenBLAS
  check_function_exists(openblas_get_num_procs BLAS_LIBRARY_IS_OPENBLAS)

  # Setup the returned variables.
  set(BLAS_IS_MKL ${BLAS_LIBRARY_IS_MKL})
  set(BLAS_IS_OPENBLAS ${BLAS_LIBRARY_IS_OPENBLAS})

  if (NOT TARGET BLAS::BLAS)
    add_library(BLAS::BLAS INTERFACE IMPORTED)
  endif ()

  target_link_libraries(BLAS::BLAS INTERFACE ${BLAS_LIBRARIES})
  target_link_options(BLAS::BLAS INTERFACE ${BLAS_LINKER_FLAGS})
endif ()

find_package_handle_standard_args(BLASImpl DEFAULT_MSG
  BLAS_LIBRARIES BLAS_FOUND BLAS_LIBRARY_HAS_REQUIRED)

mark_as_advanced(FORCE
  BLAS_LIBRARY_IS_OPENBLAS
  BLAS_MANGLING_USES_UNDERSCORE)
