h2_set_full_path(THIS_DIR_SOURCES
  runtime.cpp
)

if (H2_HAS_CUDA)
  h2_append_full_path(THIS_DIR_SOURCES runtime_cuda.cpp)
elseif (H2_HAS_ROCM)
  h2_append_full_path(THIS_DIR_SOURCES runtime_rocm.cpp)
endif ()

add_subdirectory(tensor)
add_subdirectory(cudnn)
add_subdirectory(util)
if (DISTCONV_HAS_P2P)
  add_subdirectory(p2p)
endif ()

# If +nvshmem, we want this library to build as a static library;
# otherwise, follow the rest of DiHydrogen.
if (DISTCONV_HAS_NVSHMEM)
  set(DISTCONV_LIB_TYPE STATIC)
else ()
  set(DISTCONV_LIB_TYPE)
endif ()

add_library(distconv
  ${DISTCONV_LIB_TYPE}
  ${SOURCES}
  ${THIS_DIR_SOURCES}
  ${CUDA_SOURCES}
  ${THIS_DIR_CUDA_SOURCES}
  ${DISTCONV_HEADERS})

# We don't use generator expressions for our source files, so this
# should be robust enough.
if (H2_HAS_ROCM)
  get_target_property(DISTCONV_LIB_SRCS distconv SOURCES)
  set(DISTCONV_HIP_SRCS)
  foreach (src ${DISTCONV_LIB_SRCS})
    get_filename_component(ext ${src} LAST_EXT)
    if (ext STREQUAL ".cu")
      list(APPEND DISTCONV_HIP_SRCS ${src})
    endif ()
  endforeach ()
  set_source_files_properties(${DISTCONV_HIP_SRCS} PROPERTIES LANGUAGE HIP)
endif ()

# PIC is required to link this into liblbann.so on x86
set_property(TARGET distconv PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NVSHMEM_FOUND)
  set_property(TARGET distconv PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()

target_include_directories(distconv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

message("\n\nH2_ROCM_LIBS=${H2_ROCM_LIBS}\n\n")
target_link_libraries(distconv PUBLIC
  H2Core
  ${H2_CUDA_LIBS}
  ${H2_ROCM_LIBS}
  ${Aluminum_LIBRARIES}
  MPI::MPI_CXX)

get_target_property(DISTCONV_MPI_CXX_INCL_DIRS
  MPI::MPI_CXX INTERFACE_INCLUDE_DIRECTORIES)
if (NOT DISTCONV_MPI_CXX_INCL_DIRS)
  target_include_directories(distconv PUBLIC
    $<$<COMPILE_LANGUAGE:HIP>:${MPI_CXX_COMPILER_INCLUDE_DIRS}>)
endif ()

set_target_properties(distconv
  PROPERTIES
  CXX_STANDARD 17
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)
