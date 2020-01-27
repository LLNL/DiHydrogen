# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/factory")

# Nothing to do here; just add subdirectories
add_subdirectory(factory)
add_subdirectory(multimethods)
