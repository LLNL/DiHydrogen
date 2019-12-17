find_package(Breathe 4.10.0)

if (NOT BREATHE_FOUND)
  message(WARNING
    "Sphinx has been detected, but could not find the Breathe tool. "
    "This is currently required for building the sphinx documentation. "
    "Please install breathe (e.g., \"pip install breathe\") and reconfigure.")
else ()
  h2_set_full_path(SPHINX_SOURCES
    Makefile
    conf.py
    index.rst
    requirements.txt
    )

  if (NOT SPHINX_BUILDER)
    set(SPHINX_BUILDER "html")
  endif ()
  set(SPHINX_OUTPUT_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/${SPHINX_BUILDER}")

  # Apparently the Makefile generator doesn't do this properly
  add_custom_target(sphinx-make-dir
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SPHINX_OUTPUT_DIR}"
    COMMENT "Creating the sphinx output directory."
    VERBATIM)

  set(SPHINX_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(SPHINX_BUILD_DIR "${SPHINX_OUTPUT_DIR}")

  add_custom_target(sphinx-generate-html
    COMMAND Sphinx::Sphinx "-b" "${SPHINX_BUILDER}"
    "${SPHINX_SOURCE_DIR}" "${SPHINX_BUILD_DIR}"
    BYPRODUCTS "${SPHINX_OUTPUT_DIR}/index.html"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    SOURCES "${SPHINX_SOURCES}"
    DEPENDS sphinx-make-dir
    COMMENT "Generating Sphinx documentation."
    VERBATIM)

  add_custom_target(sphinx
    COMMAND ${CMAKE_COMMAND} -E echo
    " --> Sphinx documentation built at ${SPHINX_OUTPUT_DIR}/index.html"
    DEPENDS sphinx-generate-html
    COMMENT "Sphinx documentation built at ${SPHINX_OUTPUT_DIR}/index.html"
    VERBATIM)

  add_custom_target(clean-sphinx
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${SPHINX_OUTPUT_DIR}
    COMMENT "Removing Sphinx documentation."
    VERBATIM)

  add_dependencies(doc sphinx)
  add_dependencies(clean-doc clean-sphinx)

  message(STATUS
    "Added target \"sphinx\" for generating Doxygen documentation.")
endif ()
