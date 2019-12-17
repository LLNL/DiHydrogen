add_executable(compare_binary_files compare_binary_files.cpp)
target_include_directories(compare_binary_files PUBLIC
  $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>)

add_subdirectory(tensor)
add_subdirectory(cudnn)

configure_file(test_util.sh.in
  test_util.sh @ONLY)
