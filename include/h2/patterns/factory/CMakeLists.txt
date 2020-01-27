# @H2_LICENSE_TEXT@

# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/factory")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Patterns COMPONENT PATTERNS SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  CopyFactory.hpp
  DefaultErrorPolicy.hpp
  NullptrErrorPolicy.hpp
  ObjectFactory.hpp
  PrototypeFactory.hpp
  )
