# @H2_LICENSE_TEXT@

get_filename_component(COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(COMPILER_PREFIX "${COMPILER_BIN_DIR}" DIRECTORY)

# Let the user override clang-format with its own variable. This would
# help if building with an older LLVM installation.
find_program(CLANG_FORMAT_PROGRAM clang-format
  HINTS ${CLANG_FORMAT_DIR} $ENV{CLANG_FORMAT_DIR}
  PATH_SUFFIXES bin
  DOC "The clang-format executable."
  NO_DEFAULT_PATH)

# Normal search inspired by the compiler choice. If the compiler
# happens to be, GCC, for example, users can also use LLVM_DIR. If all
# else fails, this falls back on default CMake searching.
find_program(CLANG_FORMAT_PROGRAM clang-format
  HINTS
  ${COMPILER_BIN_DIR}
  ${COMPILER_PREFIX}
  ${LLVM_DIR} $ENV{LLVM_DIR}
  PATH_SUFFIXES bin
  DOC "The clang-format executable."
  NO_DEFAULT_PATH)
find_program(CLANG_FORMAT_PROGRAM clang-format)

if (CLANG_FORMAT_PROGRAM)
  execute_process(COMMAND ${CLANG_FORMAT_PROGRAM} --version
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
    CLANG_FORMAT_VERSION
    "${CLANG_FORMAT_VERSION_STRING}")

  if (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "9.0.0")
    set(CLANG_FORMAT_VERSION_OK TRUE)
  else ()
    set(CLANG_FORMAT_VERSION_OK FALSE)
  endif ()
endif ()

if (CLANG_FORMAT_PROGRAM AND CLANG_FORMAT_VERSION_OK)
  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT_PROGRAM} -i
    $<TARGET_PROPERTY:clang-format,FORMAT_SOURCES>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Applying clang-format."
    COMMAND_EXPAND_LISTS
    VERBATIM)
  define_property(TARGET PROPERTY FORMAT_SOURCES
    BRIEF_DOCS "Sources for clang-format."
    FULL_DOCS "Sources for clang-format.")

  # Add the sources from the given target to the "clang-format"
  # target.
  macro (add_clang_format IN_TARGET)

    get_target_property(TGT_TYPE ${IN_TARGET} TYPE)

    if ((TGT_TYPE MATCHES "(STATIC|SHARED|OBJECT|MODULE)_LIBRARY")
        OR (TGT_TYPE MATCHES "EXECUTABLE"))

      message(STATUS "Adding clang-format to ${IN_TARGET}")

      unset(TGT_SOURCES_FULL_PATH)
      get_target_property(TGT_SOURCES ${IN_TARGET} SOURCES)
      get_target_property(TGT_SRC_DIR ${IN_TARGET} SOURCE_DIR)

      foreach (src IN LISTS TGT_SOURCES)
        get_filename_component(SRC_NAME "${src}" NAME)
        if (src STREQUAL SRC_NAME)
          list(APPEND TGT_SOURCES_FULL_PATH "${TGT_SRC_DIR}/${src}")
        else ()
          list(APPEND TGT_SOURCES_FULL_PATH "${src}")
        endif ()
      endforeach ()

      set_property(TARGET clang-format APPEND
        PROPERTY FORMAT_SOURCES "${TGT_SOURCES_FULL_PATH}")
    elseif (TGT_TYPE MATCHES "INTERFACE_LIBRARY")
      get_target_property(TGT_SOURCES ${IN_TARGET} INTERFACE_SOURCES)
      message("TGT_SOURCES=${TGT_SOURCES}")

      # Sources might be in generator expressions! :/ We want to only
      # change the BUILD_INTERFACE objects with absolute paths.
      foreach (src IN LISTS TGT_SOURCES)
        # Skip install files
        if (src MATCHES ".*INSTALL_INTERFACE.*")
          continue()
        endif ()

        if (src MATCHES ".*BUILD_INTERFACE:(.*)>")
          set(my_src "${CMAKE_MATCH_1}")
          message("I think the filename is ${CMAKE_MATCH_1}")
        else ()
          set(my_src "${src}")
        endif ()
        get_filename_component(SRC_NAME "${my_src}" NAME)
        # Assume a relative path is
        if (my_src STREQUAL SRC_NAME)
          message(FATAL_ERROR "AHHHH ${my_src}")
          list(APPEND TGT_SOURCES_FULL_PATH "${TGT_SRC_DIR}/${my_src}")
        else ()
          list(APPEND TGT_SOURCES_FULL_PATH "${my_src}")
        endif ()
      endforeach ()

      set_property(TARGET clang-format APPEND
        PROPERTY FORMAT_SOURCES "${TGT_SOURCES_FULL_PATH}")
    else ()
      message("Target ${IN_TARGET} is ${TGT_TYPE}")
    endif ()
  endmacro ()

  function (add_cf_to_tgts_in_dir IN_DIR)

    # Handle this directory
    get_property(_targets
      DIRECTORY "${IN_DIR}"
      PROPERTY BUILDSYSTEM_TARGETS)

    foreach (tgt IN LISTS _targets)
      add_clang_format(${tgt})
    endforeach ()

    # Recur
    get_property(_subdirs
      DIRECTORY "${IN_DIR}"
      PROPERTY SUBDIRECTORIES)

    foreach (dir IN LISTS _subdirs)
      add_cf_to_tgts_in_dir("${dir}")
    endforeach ()
  endfunction ()

  function (add_clang_format_to_all_targets)
    add_cf_to_tgts_in_dir("${CMAKE_SOURCE_DIR}")
  endfunction ()

  message(STATUS "Found clang-format: ${CLANG_FORMAT_PROGRAM} "
    "(Version: ${CLANG_FORMAT_VERSION})")
  message(STATUS
    "Added target \"clang-format\" for applying clang-format to source.")
endif ()
