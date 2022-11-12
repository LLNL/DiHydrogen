set(SOURCES
  distconv_benchmark.cpp
  shuffle_benchmark.cpp
  distconv_benchmark_pooling.cpp
  distconv_benchmark_bn.cpp)

# TODO: Generalize/rewrite to accommodate MIOpen
if (H2_HAS_CUDA)
  list(APPEND SOURCES cudnn_benchmark.cpp)
endif ()

foreach (src ${SOURCES})
  get_filename_component(target ${src} NAME_WE)
  if (${src} STREQUAL distconv_benchmark.cpp OR
      ${src} STREQUAL distconv_benchmark_pooling.cpp OR
      ${src} STREQUAL distconv_benchmark_bn.cpp)
    add_executable(${target} ${src} benchmark_common_cuda.cu)
  else ()
    add_executable(${target} ${src})
  endif ()
  target_link_libraries(${target} distconv)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17)
endforeach ()

configure_file(cudnn_benchmark_jsrun.sh.in
  cudnn_benchmark_jsrun.sh @ONLY)
