################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

if (NOT MPI_CXX_FOUND)
  find_package(MPI COMPONENT CXX REQUIRED)
endif ()
