with_nccl=OFF
if [[ -n "${gpu_arch}" ]]
then
    with_nccl=ON
fi

cmake \
    -G Ninja \
    -S ${lbann_sb_dir} \
    -B ${build_dir}/build-deps \
    \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${prefix} \
    \
    -D CMAKE_EXE_LINKER_FLAGS=${common_linker_flags} \
    -D CMAKE_SHARED_LINKER_FLAGS=${common_linker_flags} \
    \
    -D CMAKE_BUILD_RPATH=${extra_rpaths//:/|} \
    -D CMAKE_INSTALL_RPATH=${extra_rpaths//:/|} \
    \
    -D CMAKE_BUILD_RPATH_USE_ORIGIN=OFF \
    -D CMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
    -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -D CMAKE_SKIP_BUILD_RPATH=OFF \
    -D CMAKE_SKIP_INSTALL_RPATH=OFF \
    -D CMAKE_SKIP_RPATH=OFF \
    \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CMAKE_HIP_STANDARD=17 \
    \
    -D CMAKE_CUDA_ARCHITECTURES=${gpu_arch} \
    -D CMAKE_HIP_ARCHITECTURES=${gpu_arch} \
    \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
    -D LBANN_SB_DEFAULT_CUDA_OPTS=${cuda_platform} \
    -D LBANN_SB_DEFAULT_ROCM_OPTS=${rocm_platform} \
    \
    -D LBANN_SB_BUILD_hwloc=${rocm_platform} \
    -D LBANN_SB_BUILD_NCCL=${cuda_platform}
