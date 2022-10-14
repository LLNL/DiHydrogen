if (H2_HAS_CUDA)
  h2_set_full_path(THIS_DIR_SOURCES runtime_cuda.cpp)
elseif (H2_HAS_ROCM)
  h2_set_full_path(THIS_DIR_SOURCES runtime_rocm.cpp)
endif ()

h2_set_full_path(THIS_DIR_CU_SOURCES
  channel_exchange.cu
  tensor_mpi_cuda.cu
  shuffle_mpi_cuda.cu
  halo_exchange_cuda.cu
  )

if (DISTCONV_HAS_NVSHMEM)
  list(APPEND THIS_DIR_CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange_cuda_nvshmem.cu")
  list(APPEND THIS_DIR_CU_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/allreduce_nvshmem.cu")
endif ()

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
