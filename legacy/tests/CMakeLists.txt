add_executable(compare_binary_files compare_binary_files.cpp)
target_include_directories(compare_binary_files PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/legacy/include>)

add_subdirectory(tensor)
add_subdirectory(cudnn)

configure_file(test_util.sh.in
  test_util.sh @ONLY)
