h2_set_full_path(THIS_DIR_HEADERS
  config.hpp
  connection.hpp
  connection_ipc.hpp
  connection_mpi.hpp
  connection_null.hpp
  logging.hpp
  mpi.hpp
  nvtx.hpp
  p2p.hpp
  request.hpp
  util_cuda.hpp
  util.hpp
  )

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
