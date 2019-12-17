# @H2_LICENSE_TEXT@

h2_set_full_path(THIS_DIR_CXX_SOURCES
  Version.cpp)

# Subdirectories
add_subdirectory(utils)

# C++ files
add_library(H2_CXX ${CXX_SOURCES} ${THIS_DIR_CXX_SOURCES} ${H2_HEADERS})
target_include_directories(
  H2_CXX PUBLIC
  $<BUILD_INTERFACE:${CMAKE_GENERATED_INCLUDE_DIRECTORY}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_FULL_INCLUDEDIR}>)

target_link_libraries(H2_CXX PUBLIC
  BLAS::BLAS
  MPI::MPI_CXX
  $<TARGET_NAME_IF_EXISTS:Half::Half>
  $<TARGET_NAME_IF_EXISTS:OpenMP::OpenMP_CXX>
  $<TARGET_NAME_IF_EXISTS:cuda::CUDA_CXX>
  $<TARGET_NAME_IF_EXISTS:rocm::ROCm_CXX>
  )

if (H2_EXTRA_CXX_FLAGS)
  target_compile_options(H2_CXX
    PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${H2_EXTRA_CXX_FLAGS}>)
endif ()

set_target_properties(H2_CXX
  PROPERTIES
  CXX_STANDARD 14
  CXX_EXTENSIONS OFF
  CXX_STANDARD_REQUIRED ON)

# Compiler flags are not transitive in the install environment; we
# don't want to force downstreams to, e.g., compile with "-Werror"
# just because we wanted to compile that way.
target_link_libraries(H2_CXX PRIVATE
  $<BUILD_INTERFACE:$<TARGET_NAME_IF_EXISTS:h2_coverage_flags>>)

install(TARGETS H2_CXX
  EXPORT DiHydrogenTargets
  COMPONENT CXX)

list(APPEND H2_LIBRARIES H2_CXX)

# GPU
if (H2_HAS_GPU AND (GPU_SOURCES OR THIS_DIR_GPU_SOURCES))
  # FIXME: ROCm will have different flags for standards, etc, than
  # CUDA. This will need to be dealt with.
  add_library(H2_GPU ${GPU_SOURCES} ${THIS_DIR_GPU_SOURCES} ${H2_HEADERS})
  list(APPEND H2_LIBRARIES H2_GPU)

  install(TARGETS H2_GPU
    EXPORT DiHydrogenTargets
    COMPONENT GPU)
endif ()

# Propagate our targets up a level.
set(H2_LIBRARIES ${H2_LIBRARIES} PARENT_SCOPE)
