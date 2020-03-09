h2_set_full_path(THIS_DIR_SOURCES
  config.cpp
  request.cpp
  connection.cpp
  connection_null.cpp
  connection_self.cpp
  connection_ipc.cpp
  connection_mpi.cpp
  mpi.cpp
  p2p.cpp
  util_cuda.cpp
  )

h2_set_full_path(THIS_DIR_CU_SOURCES
  connection_mpi_kernels.cu
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)