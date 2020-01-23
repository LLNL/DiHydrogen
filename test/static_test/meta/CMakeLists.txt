# @H2_LICENSE_TEXT@

add_subdirectory(core)
add_subdirectory(partial_functions)
add_subdirectory(typelist)

target_sources(StaticTest
  PRIVATE
  static_test_core.cpp
  )
