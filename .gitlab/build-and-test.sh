#!/usr/bin/env bash

################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Initialize modules for users not using bash as a default shell
modules_home=${MODULESHOME:-"/usr/share/lmod/lmod"}
if [[ -e ${modules_home}/init/bash ]]
then
    source ${modules_home}/init/bash
fi

set -o errexit
set -o nounset

hostname="$(hostname)"
cluster=${hostname//[0-9]/}
project_dir="$(git rev-parse --show-toplevel)"
if [[ $? -eq 1 ]]
then
    project_dir="$(pwd)"
fi

# NOTE: No modules will be explicitly unloaded or purged. Obviously,
# loading a new compiler will trigger the auto-unload of the existing
# compiler module (and all the other side-effects wrt mpi, etc), but
# no explicit action is taken by this script.
modules=${MODULES:-""}
run_coverage=${WITH_COVERAGE:-""}
build_distconv=${WITH_DISTCONV:-""}

job_unique_id=${CI_JOB_ID:-""}
prefix=""

# Setup the module environment
if [[ -n "${modules}" ]]
then
    echo "Loading modules: \"${modules}\""
    module load ${modules}
fi

# Finish setting up the environment
source ${project_dir}/.gitlab/setup_env.sh

# Make sure our working directory is something sane.
cd ${project_dir}

# Create some temporary build space.
if [[ -z "${job_unique_id}" ]]; then
    job_unique_id=manual_job_$(date +%s)
    while [[ -d ${prefix}-${job_unique_id} ]] ; do
        sleep 1
        job_unique_id=manual_job_$(date +%s)
    done
fi
build_dir=${BUILD_DIR:-"${project_dir}/build-${job_unique_id}"}
mkdir -p ${build_dir}

# Dependencies
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test started"
echo "~~~~~         Start: $(date)"
echo "~~~~~          Host: ${hostname}"
echo "~~~~~   Project dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

prefix="${project_dir}/install-deps-${CI_JOB_NAME_SLUG:-${job_unique_id}}"

# Just for good measure...
export CMAKE_PREFIX_PATH=${prefix}/aluminum:${prefix}/catch2:${prefix}/hwloc:${prefix}/hydrogen:${prefix}/nccl:${prefix}/spdlog:${CMAKE_PREFIX_PATH}

# Allow a user to force this
rebuild_deps=${REBUILD_DEPS:-""}

# Rebuild if the prefix doesn't exist.
if [[ ! -d "${prefix}" ]]
then
    rebuild_deps=1
fi

# Rebuild if latest hashes don't match
if [[ -z "${rebuild_deps}" ]]
then
    function fetch-sha {
        # $1 is the LLNL package name (e.g., 'aluminum')
        # $2 is the branch name (e.g., 'master')
        curl -s -H "Accept: application/vnd.github.VERSION.sha" \
             "https://api.github.com/repos/llnl/$1/commits/$2"
    }

    al_head=$(fetch-sha aluminum master)
    al_prebuilt="<not found>"
    if [[ -f "${prefix}/al-prebuilt-hash.txt" ]]
    then
        al_prebuilt=$(cat ${prefix}/al-prebuilt-hash.txt)
    fi

    h_head=$(fetch-sha elemental hydrogen)
    h_prebuilt="<not found>"
    if [[ -f "${prefix}/h-prebuilt-hash.txt" ]]
    then
        h_prebuilt=$(cat ${prefix}/h-prebuilt-hash.txt)
    fi

    if [[ "${al_head}" != "${al_prebuilt}" ]]
    then
        echo "Prebuilt Aluminum hash does not match latest head; rebuilding."
        echo "  (prebuilt: ${al_prebuilt}; head: ${al_head})"
        rebuild_deps=1
    fi
    if [[ "${h_head}" != "${h_prebuilt}" ]]
    then
        echo "Prebuilt Hydrogen hash does not match latest head; rebuilding."
        echo "  (prebuilt: ${h_prebuilt}; head: ${h_head})"
        rebuild_deps=1
    fi
fi

if [[ -n "${rebuild_deps}" ]]
then

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~     Build dir: ${build_dir}"
    echo "~~~~~   Install dir: ${prefix}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Get the superbuild because why not.
    lbann_sb_top_dir=${build_dir}/sb
    lbann_sb_dir=${lbann_sb_top_dir}/scripts/superbuild
    mkdir -p ${lbann_sb_top_dir}
    cd ${lbann_sb_top_dir}

    # Sparse checkout of the SuperBuild
    git init
    git remote add origin https://github.com/llnl/lbann
    git fetch --depth=1 origin develop
    git config core.sparseCheckout true
    echo "scripts/superbuild" >> .git/info/sparse-checkout
    git pull --ff-only origin develop

    cd ${build_dir}
    # Uses "${cluster}", "${prefix}", and "${lbann_sb_dir}"
    source ${project_dir}/.gitlab/configure_deps.sh
    cmake --build build-deps

    # Stamp these commits
    cd ${build_dir}/build-deps/aluminum/src && git rev-parse HEAD > ${prefix}/al-prebuilt-hash.txt

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Dependencies Built"
    echo "~~~~~ $(date)"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
else
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Using Cached Dependencies"
    echo "~~~~~     Prefix: ${prefix}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    for f in $(find ${prefix} -iname "*.pc");
    do
        pfx=$(realpath $(dirname $(dirname $(dirname $f))))
        echo " >> Changing prefix in $(realpath $f) to: ${pfx}"
        sed -i -e "s|^prefix=.*|prefix=${pfx}|g" $f
    done
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Building DiHydrogen"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

prefix=${build_dir}/install
cd ${build_dir}
source ${project_dir}/.gitlab/configure_h2.sh
if ! cmake --build build-h2 ;
then
    echo "ERROR: compilation failed, building with verbose output..."
    cmake --build build-h2 --verbose -j 1
else
    cmake --install build-h2
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Testing DiHydrogen"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

failed_tests=0
source ${project_dir}/.gitlab/run_catch_tests.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Dihydrogen Tests Complete"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

if [[ "${run_coverage}" == "1" ]]
then

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Generating code coverage reports"
    echo "~~~~~ $(date)"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # This is beyond obnoxious
    gcovr_prefix=$(dirname $(dirname $(command -v gcovr)))
    python_path=$(ls --color=no -1 -d ${gcovr_prefix}/lib/python*/site-packages)
    echo "python_path=${python_path}"
    PYTHONPATH=${python_path}:${PYTHONPATH} cmake --build build-h2 -t coverage
    if [[ -e ${build_dir}/build-h2/coverage-gcovr.xml ]]
    then
        cp ${build_dir}/build-h2/coverage-gcovr.xml ${project_dir}
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Generated code coverage reports"
    echo "~~~~~ $(date)"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test completed"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

[[ "${failed_tests}" -eq 0 ]] && exit 0 || exit 1
