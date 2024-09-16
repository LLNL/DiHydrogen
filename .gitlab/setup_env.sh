# This is a collection of common variables and whatnot that may change
# based on the value of "${cluster}" or other variables.

# To make things work with modules, the user can set "COMPILER_FAMILY"
# to "gnu", "clang", "amdclang", or "cray" and the suitable compiler
# paths will be deduced from the current PATH. Alternatively, users
# can set "CC"/"CXX" directly, in which case the
# "COMPILER_FAMILY" variable will be ignored.

compiler_family=${COMPILER_FAMILY:-gnu}
case "${compiler_family,,}" in
    gnu|gcc)
        CC=${CC:-$(command -v gcc)}
        CXX=${CXX:-$(command -v g++)}
        ;;
    clang)
        CC=${CC:-$(command -v clang)}
        CXX=${CXX:-$(command -v clang++)}
        ;;
    amdclang)
        CC=${CC:-$(command -v amdclang)}
        CXX=${CXX:-$(command -v amdclang++)}
        ;;
    cray)
        CC=${CC:-$(command -v cc)}
        CXX=${CXX:-$(command -v CC)}
        ;;
    *)
        echo "Unknown compiler family: ${compiler_family}. Using gnu."
        CC=${CC:-$(command -v gcc)}
        CXX=${CXX:-$(command -v g++)}
        ;;
esac

# HIP/CUDA configuration and launcher are platform-specific
CUDACXX=${CUDACXX:=""}
CUDAHOSTCXX=${CUDAHOSTCXX:=${CXX}}

cuda_platform=OFF
rocm_platform=OFF

launcher=mpiexec

common_linker_flags="-Wl,--disable-new-dtags"
extra_rpaths=${extra_rpaths:-""}

case "${cluster}" in
    pascal)
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=60
        launcher=slurm
        ;;
    lassen)
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=70
        launcher=lsf
        ;;
    tioga)
        cray_libs_dir=${CRAYLIBS_X86_64:-""}
        if [[ -n "${cray_libs_dir}" ]]
        then
            extra_rpaths="${cray_libs_dir}:${ROCM_PATH}/lib:${extra_rpaths}"
        else
            extra_rpaths="${ROCM_PATH}/lib:${extra_rpaths}"
        fi
        rocm_platform=ON
	gpu_arch=gfx90a
        launcher=flux
        ;;
    corona)
        # Only turn on GPU stuff if ROCm module has been loaded, which
        # is checked by testing for ROCM_PATH.
        extra_rpaths=${ROCM_PATH:+${ROCM_PATH}/lib:${extra_rpaths}}
	gpu_arch=${ROCM_PATH:+gfx906}
        if [[ -n "${gpu_arch}" ]]; then
            rocm_platform=ON
        fi
        launcher=flux
        ;;
    *)
        ;;
esac

CFLAGS=${CFLAGS:-""}
CXXFLAGS=${CXXFLAGS:-""}
LDFLAGS=${LDFLAGS:-""}
LDFLAGS="${common_linker_flags} ${LDFLAGS}"

# Make sure the compilers and flags are exported
export CC CXX CUDACXX CUDAHOSTCXX CFLAGS CXXFLAGS LDFLAGS
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Environment Info"
echo "~~~~~"
echo "~~~~~  Cluster: ${cluster}"
echo "~~~~~  CUDA? ${cuda_platform}"
echo "~~~~~  ROCm? ${rocm_platform}"
echo "~~~~~  GPU arch: ${gpu_arch}"
echo "~~~~~  Launcher: ${launcher}"
echo "~~~~~"
echo "~~~~~  Compiler family: ${compiler_family}"
echo "~~~~~  CC: ${CC}"
echo "~~~~~  CXX: ${CXX}"
echo "~~~~~  CUDACXX: ${CUDACXX}"
echo "~~~~~  CUDAHOSTCXX: ${CUDAHOSTCXX}"
echo "~~~~~"
echo "~~~~~  CFLAGS: ${CFLAGS}"
echo "~~~~~  CXXFLAGS: ${CXXFLAGS}"
echo "~~~~~  LDFLAGS: ${LDFLAGS}"
echo "~~~~~  Extra rpaths: ${extra_rpaths}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Handle cuDNN
if [[ "${cuda_platform}" == "ON" ]]
then
    cuda_maj_version=$(basename ${CUDA_HOME} | grep -E --color=no -o "[0-9]+\.[0-9]+\.[0-9]+" | cut -d '.' -f 1)
    arch=$(uname -m)
    cudnn_root=$(ls -1 -d /usr/workspace/brain/cudnn/cudnn-*/cuda_${cuda_maj_version}_${arch} | tail -1)
    if [[ -z "${cudnn_root}" ]]
    then
        echo "WARNING: No suitable cuDNN found."
    else
        CMAKE_PREFIX_PATH=${cudnn_root}:${CMAKE_PREFIX_PATH:-""}
    fi
fi

# Get Breathe, gcovr, and Ninja. Putting this off to the side because
# I don't want to tweak "the real" python environment, but it's just
# these one or two things so it's not worth a venv.
if [[ -n "${run_coverage}" ]]
then
    python_pkgs="ninja gcovr"
else
    python_pkgs="ninja"
fi

export PYTHONUSERBASE=${TMPDIR}/${USER}/python/${cluster}
export PATH=${PYTHONUSERBASE}/bin:${PATH}
python3 -m pip install --user ${python_pkgs}

# Make sure the PYTHONPATH is all good.
export PYTHONPATH=$(ls --color=no -1 -d ${PYTHONUSERBASE}/lib/python*/site-packages | paste -sd ":" - ):${PYTHONPATH:-""}
