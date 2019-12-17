# @H2_LICENSE_TEXT@

h2_set_full_path(THIS_DIR_HEADERS
  Error.hpp
  )

# Propagate headers up the tree
set(HEADERS ${HEADERS} ${THIS_DIR_HEADERS} PARENT_SCOPE)
