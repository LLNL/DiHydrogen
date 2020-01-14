h2_set_full_path(THIS_DIR_HEADERS
  backend.hpp
  batchnorm.hpp
  convolution.hpp
  pooling.hpp
  relu.hpp
  leaky_relu.hpp
  deconvolution.hpp
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)