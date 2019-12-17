set_full_path(THIS_DIR_HEADERS
  stopwatch.h
  util.hpp
  util_cudnn.hpp
  util_mpi.hpp
  util_cuda.hpp  
  cxxopts.hpp
  )

if (DISTCONV_HAS_NVSHMEM)
  list(APPEND THIS_DIR_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/nvshmem.hpp")
endif ()

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)