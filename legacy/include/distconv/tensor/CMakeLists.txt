h2_set_full_path(THIS_DIR_HEADERS
  algorithms_cuda.hpp
  algorithms.hpp
  channel_exchange.hpp
  distribution.hpp
  halo_cuda.hpp
  halo_exchange_cuda.hpp
  halo_exchange_cuda_mpi.hpp
  halo_exchange_cuda_al.hpp
  halo_exchange.hpp
  halo_packing_cuda.hpp
  memory_cuda.hpp
  memory.hpp
  runtime_cuda.hpp
  runtime.hpp
  shuffle_mpi.hpp
  shuffle_mpi_cuda.hpp
  shuffle_mpi_cuda_al.hpp
  stream.hpp
  stream_cuda.hpp
  tensor_base.hpp
  tensor_cuda.hpp
  tensor.hpp
  tensor_mpi_cuda.hpp
  tensor_mpi.hpp
  tensor_process.hpp
  allreduce.hpp
  allreduce_mpi.hpp
  allreduce_mpi_cuda.hpp
  allreduce_al.hpp
  )

if (DISTCONV_HAS_P2P)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange_cuda_p2p.hpp")
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange_cuda_hybrid.hpp")
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shuffle_mpi_cuda_p2p.hpp")
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/shuffle_mpi_cuda_hybrid.hpp")
endif ()

if (DISTCONV_HAS_NVSHMEM)
  list(APPEND THIS_DIR_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/halo_exchange_cuda_nvshmem.hpp")
endif ()

add_subdirectory(algorithms)

set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
