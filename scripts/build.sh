#!/usr/bin/env bash

set -u
set -e

#######################
SYSTEM=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
# Compiler paths
COMPILER=${COMPILER:-gnu}
case $COMPILER in
    xl)
        C_COMPILER=xlc
        CXX_COMPILER=xlc++
        Fortran_COMPILER=xlf
        OPENMP_FLAG=-qsmp=omp
        ;;
    clang)
        C_COMPILER=clang
        CXX_COMPILER=clang++
        OPENMP_FLAG=-fopenmp
        ;;
    gnu)
        C_COMPILER=gcc
        CXX_COMPILER=g++
        Fortran_COMPILER=gfortran
        OPENMP_FLAG=-fopenmp
        ;;
esac
MPI_C_COMPILER=mpicc
MPI_CXX_COMPILER=mpicxx
MPI_Fortran_COMPILER=mpifort
MPI_HOME=$(dirname $(dirname $(type $MPI_C_COMPILER | awk '{print $3}')))
if [[ $MPI_HOME == *spectrum* ]]; then
    MPI=smpi
elif [[ $MPI_HOME == *mvapich* ]]; then
    MPI=mvapich
else
    MPI=unknown
fi
CUDA_VERSION=$(nvcc --version | grep -oE "V[0-9]+\.[0-9]+\.[0-9]+" | sed 's/V//')
CUDA_VERSION_MAJOR=$(echo $CUDA_VERSION | sed 's/\(.\+\)\.\(.\+\)\.\(.\+\)/\1/')
CUDA_VERSION_MINOR=$(echo $CUDA_VERSION | sed 's/\(.\+\)\.\(.\+\)\.\(.\+\)/\2/')

# Build paths
DEFAULT_BUILD_TYPE=Release
BUILD_TYPE=${BUILD_TYPE:-$DEFAULT_BUILD_TYPE}
TOP_DIR=$(realpath $(dirname $0)/..)
SRC_DIR=$TOP_DIR
UNIQUE_PATH=${SYSTEM}/${COMPILER}_${MPI}_cuda-${CUDA_VERSION}/${BUILD_TYPE}
BUILD_DIR=${PWD}/build/$UNIQUE_PATH
INSTALL_DIR=${PWD}/install/$UNIQUE_PATH

# Build options (off when undefined)
ENABLE_NVSHMEM=
ENABLE_P2P=

# External library paths
Aluminum_DIR=${Aluminum_DIR:-$HOME/wsa/aluminum/install/$UNIQUE_PATH}
CUB_DIR=${CUB_DIR:-/usr/workspace/wsb/brain/cub/cub-1.8.0}
NVSHMEM_VERSION=0.3.3
NVSHMEM_DIR=${NVSHMEM_DIR:-/usr/workspace/wsb/brain/nvshmem/nvshmem_$NVSHMEM_VERSION/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}_$(uname -p)}
P2P_DIR=${P2P_DIR:-$HOME/wsa/p2p/install/$UNIQUE_PATH}

function print_usage() {
    cat <<EOM
$(basename $0) ARGS

where ARGS are:
    --with-nvshmem [DIR]    Enable NVSHMEM using one installed at DIR
    --with-p2p [DIR]        Enable P2P using one installed at DIR
    --help                  Display help message
EOM
}

OPTS=$(getopt -o h -l help,with-al::,with-p2p::,with-nvshmem:: -n $0 -- "$@")
if [ $? != 0 ]; then echo Failed parsing options.; exit 1; fi
eval set -- $OPTS
while true; do
    case $1 in
        --with-al)
            if [[ $2 ]]; then
                Aluminum_DIR=$2
            fi
            shift 2
            ;;
        --with-nvshmem)
            ENABLE_NVSHMEM=1
            if [[ $2 ]]; then
                NVSHMEM_DIR=$2
            fi
            shift 2
            ;;
        --with-p2p)
            ENABLE_P2P=1
            if [[ $2 ]]; then
                P2P_DIR=$2
            fi
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        --)
            shift; break
            ;;
        *)
            echo "Invalid option: $1" >&2
            print_usage >&2
            exit 1
            ;;
    esac
done

if [[ ! -d $Aluminum_DIR ]]; then
    echo "Error! Aluminum not found (Aluminum_DIR: $Aluminum_DIR)"
    exit 1
fi
if [[ $ENABLE_P2P ]]; then
    if [[ ! -d $P2P_DIR ]]; then
        echo "Error! P2P requested but not found (P2P_DIR: $P2P_DIR)"
        exit 1
    fi
    echo "P2P enabled with $P2P_DIR"
else
    P2P_DIR=
    echo "P2P disabled"
fi
if [[ $ENABLE_NVSHMEM ]]; then
    if [[ ! -d $NVSHMEM_DIR ]]; then
        echo "Error! NVSHMEM requested but not found (NVSHMEM_DIR: $NVSHMEM_DIR)"
        exit 1
    fi
    echo "NVSHMEM enabled with $NVSHMEM_DIR"
else
    NVSHMEM_DIR=
    echo "NVSHMEM disabled"
fi

# cmake
mkdir -p $BUILD_DIR
pushd $BUILD_DIR
cmake ${SRC_DIR} \
      -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR \
      -DCMAKE_BUILD_TYPE=$BUILD_TYPE \
      -DCMAKE_C_COMPILER=$C_COMPILER \
      -DCMAKE_CXX_COMPILER=$CXX_COMPILER \
      -DCMAKE_CUDA_HOST_COMPILER=$(which $CXX_COMPILER) \
      -DH2_ENABLE_CUDA=ON \
      -DAluminum_DIR=$Aluminum_DIR \
      -DP2P_DIR=$P2P_DIR \
      -DNVSHMEM_DIR=$NVSHMEM_DIR \
      -DCUB_DIR=$CUB_DIR \
      -DCatch2_DIR=/usr/workspace/wsb/brain/catch2/lib64/cmake/Catch2 \
      -DDISTCONV_OPTIMIZE_FIND_DESTINATION=ON \
      -G Ninja

if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "CMAKE FAILED"
    echo "--------------------"
	popd
    exit 1
fi

# build
BUILD_COMMAND="ninja -v"
echo "$BUILD_COMMAND"
${BUILD_COMMAND}
if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "BUILD FAILED"
    echo "--------------------"
    exit 1
fi

# Install
INSTALL_COMMAND=$(cat << EOF
ninja install
EOF
)
echo "${INSTALL_COMMAND}"
${INSTALL_COMMAND}
if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "INSTALL FAILED"
    echo "--------------------"
    exit 1
fi

popd
