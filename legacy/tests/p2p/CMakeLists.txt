
set(TEST_SOURCES
  test_sendrecv.cpp
  test_put.cpp
  test_ring.cpp
  test_bandwidth_neighbor.cu
  benchmark_pingpong_sendrecv.cu
  benchmark_exchange_ring.cu
  benchmark_pingpong_put.cpp)

foreach (src ${TEST_SOURCES})
  get_filename_component(target ${src} NAME_WE)
  add_executable(${target} ${src} test_util_cuda.cu)
  target_link_libraries(${target} distconv)
  target_include_directories(${target} PUBLIC ..)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 14)
endforeach ()
