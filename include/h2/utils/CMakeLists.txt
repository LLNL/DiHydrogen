# @H2_LICENSE_TEXT@

# Setup the base prefix for all the includes
set(H2_CURRENT_INSTALL_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/utils")

h2_add_sources_to_target_and_install(
  TARGET H2Core COMPONENT CORE SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Error.hpp
  )
