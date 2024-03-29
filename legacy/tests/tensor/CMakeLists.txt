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
  test_halo_exchange_cuda.cu
  test_concat_mpi_cuda.cu
  test_allreduce_cuda.cu)

foreach (src ${TEST_SOURCES})
  get_filename_component(target ${src} NAME_WE)
  get_filename_component(ext ${src} LAST_EXT)

  add_executable(${target} ${src})
  target_link_libraries(${target} distconv H2Core)

  if (ext STREQUAL ".cu")
    if (H2_HAS_CUDA)
      set_target_properties(${target} PROPERTIES CXX_STANDARD 17)
    else ()
      set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
      set_target_properties(${target} PROPERTIES HIP_STANDARD 17)
    endif ()
  else ()
    set_target_properties(${target} PROPERTIES CXX_STANDARD 17)
  endif ()
  target_compile_options(${target} PRIVATE "-Wno-unused-result")
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
