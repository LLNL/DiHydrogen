# This is a collection of common variables and whatnot that may change
# based on the value of "${cluster}" or other variables.

# Users can set compilers using the documented CMake environment
# variables.
CC=${CC:=""}
CXX=${CXX:=""}
CUDACXX=${CUDACXX:=""}
CUDAHOSTCXX=${CUDAHOSTCXX:=${CXX}}

cuda_platform=OFF
rocm_platform=OFF

launcher=mpiexec

common_linker_flags="-Wl,--disable-new-dtags"
extra_rpaths=${extra_rpaths:-""}

case "${cluster}" in
    pascal)
        CC=${CC:-$(command -v gcc)}
        CXX=${CXX:-$(command -v g++)}
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=60
        launcher=slurm
        ;;
    lassen)
        CC=${CC:-$(command -v clang)}
        CXX=${CXX:-$(command -v clang++)}
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=70
        launcher=lsf
        ;;
    tioga)
        CC=${CC:-$(command -v amdclang)}
        CXX=${CXX:-$(command -v amdclang++)}
        if [[ -n "${CRAYLIBS_X86_64}" ]]
        then
            extra_rpaths="${CRAYLIBS_X86_64}:${ROCM_PATH}/lib:${extra_rpaths}"
        else
            extra_rpaths="${ROCM_PATH}/lib:${extra_rpaths}"
        fi
        rocm_platform=ON
	gpu_arch=gfx90a
        launcher=flux
        ;;
    corona)
        CC=${CC:-$(command -v amdclang)}
        CXX=${CXX:-$(command -v amdclang++)}
        CUDACXX=""
        extra_rpaths="${ROCM_PATH}/lib:${extra_rpaths}"
        rocm_platform=ON
	gpu_arch=gfx906
        launcher=flux
        ;;
    *)
        ;;
esac

# Make sure the compilers are exported
LDFLAGS=${LDFLAGS:-""}
LDFLAGS="${common_linker_flags} ${LDFLAGS}"
export CC CXX CUDACXX CUDAHOSTCXX LDFLAGS

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

# A bit of added safety...
export CMAKE_PREFIX_PATH=${prefix}/aluminum:${prefix}/catch2:${prefix}/hwloc:${prefix}/hydrogen:${prefix}/nccl:${prefix}/spdlog:${CMAKE_PREFIX_PATH}

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
