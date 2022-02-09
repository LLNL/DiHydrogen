h2_set_full_path(THIS_DIR_SOURCES
  runtime.cpp
  runtime_cuda.cpp
  )

add_subdirectory(tensor)
add_subdirectory(cudnn)
add_subdirectory(util)
if (DISTCONV_HAS_P2P)
  add_subdirectory(p2p)
endif ()

add_library(distconv
  ${SOURCES} ${THIS_DIR_SOURCES}
  ${CUDA_SOURCES} ${THIS_DIR_CUDA_SOURCES}
  ${DISTCONV_HEADERS})

# PIC is required to link this into liblbann.so on x86
set_property(TARGET distconv PROPERTY POSITION_INDEPENDENT_CODE ON)

if (NVSHMEM_FOUND)
  set_property(TARGET distconv PROPERTY CUDA_SEPARABLE_COMPILATION ON)
endif ()

target_include_directories(distconv PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>)

target_link_libraries(distconv PUBLIC MPI::MPI_CXX)
target_link_libraries(distconv PUBLIC h2::cuda_toolkit)
target_link_libraries(distconv PUBLIC ${Aluminum_LIBRARIES})
if (NVSHMEM_FOUND)
  target_link_libraries(distconv PUBLIC NVSHMEM::NVSHMEM)
endif ()

set_target_properties(distconv
  PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)
