add_executable(compare_binary_files compare_binary_files.cpp)
target_include_directories(compare_binary_files PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/legacy/include>)
set_target_properties(compare_binary_files
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

add_subdirectory(tensor)
add_subdirectory(dnn_backend)
if (DISTCONV_HAS_P2P)
  add_subdirectory(p2p)
endif ()

configure_file(test_util.sh.in
  test_util.sh @ONLY)
