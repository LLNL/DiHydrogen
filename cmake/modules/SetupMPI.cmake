################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# This module configures MPI and ensures the library is setup properly

if (NOT MPI_CXX_FOUND)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif ()

if (NOT TARGET MPI::MPI_CXX)
  add_library(MPI::MPI_CXX INTERFACE IMPORTED)
  if (MPI_CXX_COMPILE_FLAGS)
    separate_arguments(_MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND
      "${MPI_CXX_COMPILE_FLAGS}")
    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
  endif()

  if (MPI_CXX_LINK_FLAGS)
    separate_arguments(_MPI_CXX_LINK_LINE UNIX_COMMAND
      "${MPI_CXX_LINK_FLAGS}")
  endif()

  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_LINK_LIBRARIES "${_MPI_CXX_LINK_LINE}")

  set_property(TARGET MPI::MPI_CXX APPEND PROPERTY
    LINK_FLAGS "${_MPI_CXX_LINK_LINE}")

  set_property(TARGET MPI::MPI_CXX PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")

endif (NOT TARGET MPI::MPI_CXX)

# Patch around pthread on Lassen
get_property(_TMP_MPI_CXX_COMPILE_FLAGS TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_COMPILE_OPTIONS)
set_property(TARGET MPI::MPI_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS
  $<$<COMPILE_LANGUAGE:CXX>:${_TMP_MPI_CXX_COMPILE_FLAGS}>)

get_property(_TMP_MPI_LINK_LIBRARIES TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES)
foreach(lib IN LISTS _TMP_MPI_LINK_LIBRARIES)
  if ("${lib}" MATCHES "-Wl*")
    list(APPEND _MPI_LINK_FLAGS "${lib}")
  else()
    list(APPEND _MPI_LINK_LIBRARIES "${lib}")
  endif ()
endforeach()

#set_property(TARGET MPI::MPI_CXX PROPERTY LINK_FLAGS ${_MPI_LINK_FLAGS})
set_property(TARGET MPI::MPI_CXX
  PROPERTY INTERFACE_LINK_LIBRARIES ${_MPI_LINK_LIBRARIES})
