if (H2_HAS_CUDA)
  h2_set_full_path(THIS_DIR_SOURCES backend.cpp)
elseif (H2_HAS_ROCM)
  h2_set_full_path(THIS_DIR_SOURCES backend_miopen.cpp)
endif ()

h2_set_full_path(THIS_DIR_CU_SOURCES
  pooling.cu
  batchnorm.cu
  leaky_relu.cu
  mean_squared_error.cu
  softmax.cu
  cross_entropy.cu
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
