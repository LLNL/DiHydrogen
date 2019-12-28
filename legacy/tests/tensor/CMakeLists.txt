set(TEST_SOURCES
  test_tensor.cpp
  test_tensor_mpi.cpp
  test_tensor_mpi_copy.cpp
  test_tensor_cuda.cu
  test_tensor_mpi_cuda.cu
  test_tensor_mpi_cuda_copy.cu
  test_tensor_mpi_cuda_algorithms.cu
  test_tensor_mpi_shuffle.cpp
  test_tensor_mpi_cuda_shuffle.cu
  test_halo_exchange_cuda.cu)

foreach (src ${TEST_SOURCES})
  get_filename_component(target ${src} NAME_WE)
  add_executable(${target} ${src})
  target_link_libraries(${target} distconv)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 14)
endforeach ()

configure_file(test_tensor.sh.in
  test_tensor.sh @ONLY)
configure_file(test_tensor_cuda.sh.in
  test_tensor_cuda.sh @ONLY)
configure_file(test_tensor_mpi.sh.in
  test_tensor_mpi.sh @ONLY)
configure_file(test_tensor_mpi_shuffle.sh.in
  test_tensor_mpi_shuffle.sh @ONLY)
configure_file(test_tensor_mpi_cuda_copy.sh.in
  test_tensor_mpi_cuda_copy.sh @ONLY)
configure_file(test_halo_exchange_cuda.sh.in
  test_halo_exchange_cuda.sh @ONLY)
configure_file(test_tensor_mpi_cuda_algorithms.sh.in
  test_tensor_mpi_cuda_algorithms.sh @ONLY)