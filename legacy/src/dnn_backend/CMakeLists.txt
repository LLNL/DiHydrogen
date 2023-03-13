if (H2_HAS_CUDA)
  h2_set_full_path(THIS_DIR_SOURCES backend.cpp pack_unpack.cpp)
elseif (H2_HAS_ROCM)
  h2_set_full_path(THIS_DIR_SOURCES backend_miopen.cpp pack_unpack.cpp)
endif ()

if (H2_HAS_DACE)
  h2_set_full_path(THIS_DIR_DACE_SOURCES backend_dace.cpp)
endif ()

h2_set_full_path(THIS_DIR_CU_SOURCES
  pooling.cu
  batchnorm.cu
  leaky_relu.cu
  mean_squared_error.cu
  softmax.cu
  cross_entropy.cu
)

if (H2_HAS_ROCM)
  h2_set_full_path(PACK_UNPACK_KERNEL_FILE pack_unpack.cu)
  list(APPEND THIS_DIR_CU_SOURCES "${PACK_UNPACK_KERNEL_FILE}")
endif ()

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" "${THIS_DIR_DACE_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
