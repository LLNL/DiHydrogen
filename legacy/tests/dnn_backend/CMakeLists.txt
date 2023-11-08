set(TEST_SOURCES
  test_leaky_relu.cpp)

foreach (src ${TEST_SOURCES})
  get_filename_component(target ${src} NAME_WE)
  add_executable(${target} ${src})
  target_link_libraries(${target} distconv)
  target_include_directories(${target} PUBLIC ..)
  set_target_properties(${target} PROPERTIES CXX_STANDARD 17)
endforeach ()

configure_file(test_cudnn_util.sh.in test_cudnn_util.sh @ONLY)
configure_file(test_distconv.sh.in test_distconv.sh @ONLY)
configure_file(test_convolution_cosmoflow.sh.in test_convolution_cosmoflow.sh @ONLY)
configure_file(test_pooling.sh.in test_pooling.sh @ONLY)
configure_file(chainer_pooling.py chainer_pooling.py COPYONLY)
