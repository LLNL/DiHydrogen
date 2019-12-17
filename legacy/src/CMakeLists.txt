set_full_path(THIS_DIR_SOURCES
  runtime.cpp
  runtime_cuda.cpp
  )

add_subdirectory(tensor)
add_subdirectory(cudnn)
add_subdirectory(util)

set(DISTCONV_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(DISTCONV_SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(DISTCONV_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
