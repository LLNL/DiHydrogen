set_full_path(THIS_DIR_HEADERS
  common_cuda.hpp
  common.hpp
  reduce_sum_cuda.hpp
  reduce_sum.hpp
  transform_cuda.hpp
  transform.hpp
  transform_reduce_sum_cuda.hpp
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)