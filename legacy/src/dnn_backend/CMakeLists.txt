h2_set_full_path(THIS_DIR_SOURCES
  communicator_manager.cpp
  dnn_backend.cpp
  options.cpp
  pack_unpack.cpp
  stream_manager.cpp
)

h2_set_full_path(THIS_DIR_CU_SOURCES
  batchnorm.cu
  cross_entropy.cu
  leaky_relu.cu
  mean_squared_error.cu
  pack_unpack.cu
  pooling.cu
  softmax.cu
)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" "${THIS_DIR_SOURCES}"
  PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
