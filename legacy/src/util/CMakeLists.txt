h2_set_full_path(THIS_DIR_SOURCES
  util.cpp
)
if (H2_HAS_CUDA)
  h2_append_full_path(THIS_DIR_SOURCES util_cuda.cpp)
elseif (H2_HAS_ROCM)
  h2_append_full_path(THIS_DIR_SOURCES util_rocm.cpp)
endif ()

if (DISTCONV_HAS_NVSHMEM)
  h2_set_full_path(THIS_DIR_CU_SOURCES
    nvshmem.cu
    )
endif ()

set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
