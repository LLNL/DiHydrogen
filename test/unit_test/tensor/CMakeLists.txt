################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

target_sources(SeqCatchTests PRIVATE
  unit_test_raw_buffer.cpp
  unit_test_strided_memory.cpp
  unit_test_tensor.cpp
  unit_test_hydrogen_interop.cpp
)

if (H2_HAS_GPU)
  target_sources(GPUCatchTests PRIVATE
    unit_test_raw_buffer.cpp
    unit_test_strided_memory.cpp
    unit_test_tensor.cpp
    unit_test_hydrogen_interop.cpp
  )
endif ()
