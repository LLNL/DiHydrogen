h2_set_full_path(THIS_DIR_HEADERS
  base.hpp
  distconv.hpp
  runtime.hpp
  runtime_cuda.hpp
  )

add_subdirectory(util)
add_subdirectory(tensor)
add_subdirectory(ref)
add_subdirectory(dnn_backend)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
