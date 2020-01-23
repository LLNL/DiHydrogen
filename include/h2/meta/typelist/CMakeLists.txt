# @H2_LICENSE_TEXT@

# Append this directory to the current install prefix
set(H2_CURRENT_INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}/typelist")

# Setup this directory's files
h2_add_sources_to_target_and_install(
  TARGET H2Meta COMPONENT META SCOPE INTERFACE
  INSTALL_PREFIX "${H2_CURRENT_INSTALL_PREFIX}"
  SOURCES
  Append.hpp
  At.hpp
  Expand.hpp
  Find.hpp
  HaskellAccessors.hpp
  Length.hpp
  LispAccessors.hpp
  Member.hpp
  Remove.hpp
  RemoveAll.hpp
  Replace.hpp
  ReplaceAll.hpp
#  Select.hpp
#  SelectAll.hpp
#  Sort.hpp
  TypeList.hpp
  Unique.hpp
  )
