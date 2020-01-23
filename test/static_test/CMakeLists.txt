# @H2_LICENSE_TEXT@

# This library might be empty; definitely shouldn't provide useful
# symbols -- it's really just a way to "run the tests" for static
# testing. As such, it is not installed and it is not exported in any
# way.
add_library(StaticTest MODULE "")
add_subdirectory(meta)
target_link_libraries(StaticTest PRIVATE H2Meta)
