# @H2_LICENSE_TEXT@

# Setup the base prefix for all the includes
set(H2_CURRENT_INSTALL_PREFIX "${CMAKE_INSTALL_INCLUDEDIR}/h2")

h2_add_sources_to_target_and_install(
  TARGET H2Core
  COMPONENT CORE
  SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Version.hpp
  )

# Setup the metaprogramming component. The metaprogramming facilities
# are expected to be available everywhere and carry no
# intra-DiHydrogen dependencies -- there is only an external
# dependence on the standard library <utility> header as
# `std::forward`, `std::move`, and `std::declval` have not been
# implemented independently.
# Create the metaprogramming library
add_library(H2Meta INTERFACE)

# Populate the sources
add_subdirectory(meta)

# All DiHydrogen linkage is package/component-scoped, e.g.,
# <h2/meta/TypeList.hpp>.
target_include_directories(H2Meta INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )

# Export the metaprogramming target
install(TARGETS H2Meta
  EXPORT ${PROJECT_NAME}Targets
  COMPONENT META
  )

# Some additional utilities.
# WARNING: Deprecated. Will be moved to a different target.
add_subdirectory(utils)
