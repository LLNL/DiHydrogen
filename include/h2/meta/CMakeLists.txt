# @H2_LICENSE_TEXT@

# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/meta")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Meta COMPONENT META SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Core.hpp
  PartialFunctions.hpp
  TypeList.hpp
  )

# Get other directories' files
add_subdirectory(core)
add_subdirectory(partial_functions)
add_subdirectory(typelist)
