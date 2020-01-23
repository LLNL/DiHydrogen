# @H2_LICENSE_TEXT@

# First add this directory's sources. CPP files aren't exported, so
# these are PRIVATE.
target_sources(H2Core PRIVATE Version.cpp)

# Subdirectories
add_subdirectory(utils)
