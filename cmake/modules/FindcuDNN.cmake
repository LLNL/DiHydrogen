# @H2_LICENSE_TEXT@

#[=============[.rst
FindcuDNN
==========

Finds the cuDNN library.

The following variables will be defined::

  cuDNN_FOUND          - True if the system has the cuDNN library.
  cuDNN_INCLUDE_DIRS   - The include directory needed for cuDNN.
  cuDNN_LIBRARIES      - The libraries needed for cuDNN.

The following cache variable will be set and marked as "advanced"::

  cuDNN_INCLUDE_PATH - The include directory needed for cuDNN.
  cuDNN_LIBRARY      - The library needed for cuDNN.

In addition, the :prop_tgt:`IMPORTED` target ``cuda::cuDNN`` will
be created.

#]=============]

find_path(cuDNN_INCLUDE_PATH cudnn.h
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} $ENV{CUDA_TOOLKIT_ROOT_DIR}
  ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "Location of cudnn header."
  )
find_path(cuDNN_INCLUDE_PATH cudnn.h)

find_library(cuDNN_LIBRARY cudnn
  HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR} ${cuDNN_DIR} $ENV{cuDNN_DIR}
  ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib
  NO_DEFAULT_PATH
  DOC "The cudnn library."
  )
find_library(cuDNN_LIBRARY cudnn)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cuDNN
  DEFAULT_MSG cuDNN_LIBRARY cuDNN_INCLUDE_PATH)

if (NOT TARGET cuda::cuDNN)
  add_library(cuda::cuDNN INTERFACE IMPORTED)
endif (NOT TARGET cuda::cuDNN)

target_include_directories(cuda::cuDNN INTERFACE "${cuDNN_INCLUDE_PATH}")
target_link_libraries(cuda::cuDNN INTERFACE "${cuDNN_LIBRARY}")

set(cuDNN_INCLUDE_DIRS "${cuDNN_INCLUDE_PATH}")
set(cuDNN_LIBRARIES cuda::cuDNN)

mark_as_advanced(FORCE
  cuDNN_INCLUDE_PATH
  cuDNN_LIBRARY)
