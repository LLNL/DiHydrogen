if [[ "$cluster" == "lassen" ]]
then
    lapack_opt="-D BLA_VENDOR=Generic"
else
    lapack_opt=""
fi

cmake -G Ninja \
      -S ${project_dir} \
      -B ${build_dir}/build-h2 \
      \
      -D CMAKE_BUILD_TYPE=Debug \
      -D CMAKE_INSTALL_PREFIX=${prefix}/dihydrogen \
      \
      -D CMAKE_BUILD_RPATH="${extra_rpaths//:/\;}" \
      -D CMAKE_INSTALL_RPATH="${extra_rpaths//:/\;}" \
      \
      -D H2_ENABLE_CUDA=${cuda_platform} \
      -D CMAKE_CUDA_ARCHITECTURES=${gpu_arch} \
      \
      -D H2_ENABLE_ROCM=${rocm_platform} \
      -D CMAKE_HIP_ARCHITECTURES=${gpu_arch} \
      -D AMDGPU_TARGETS=${gpu_arch} \
      -D GPU_TARGETS=${gpu_arch} \
      \
      ${lapack_opt} \
      -D H2_CI_BUILD=${run_coverage:-OFF} \
      -D H2_DEVELOPER_BUILD=ON \
      -D H2_ENABLE_CODE_COVERAGE=${run_coverage:-OFF} \
      -D H2_ENABLE_DISTCONV_LEGACY=${build_distconv:-OFF}
