add_subdirectory(distconv)
if (DISTCONV_HAS_P2P)
  add_subdirectory(p2p)
endif ()

set(DISTCONV_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)