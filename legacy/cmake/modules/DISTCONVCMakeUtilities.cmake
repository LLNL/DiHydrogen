# Copied from LBANN

# A handy macro to add the current source directory to a local
# filename. To be used for creating a list of sources.
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()
