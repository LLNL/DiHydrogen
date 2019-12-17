# Proper C++ files to add to the library
h2_set_full_path(THIS_DIR_CXX_SOURCES
  Error.cpp)

set(CXX_SOURCES ${CXX_SOURCES} ${THIS_DIR_CXX_SOURCES} PARENT_SCOPE)
