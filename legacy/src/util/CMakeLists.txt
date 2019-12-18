h2_set_full_path(THIS_DIR_SOURCES
  util.cpp
  util_cuda.cpp
  )

if (DISTCONV_HAS_NVSHMEM)
  h2_set_full_path(THIS_DIR_CU_SOURCES
    nvshmem.cu
    )
endif ()

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)