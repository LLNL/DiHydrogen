h2_set_full_path(THIS_DIR_HEADERS
  Version.hpp
  )

add_subdirectory(utils)

# Install the headers
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING PATTERN "*.hpp")

# Propagate headers up the tree
set(H2_HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
