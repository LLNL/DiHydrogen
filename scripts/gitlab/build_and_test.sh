#!/usr/bin/env bash

################################################################################
## Copyright 2019-2023 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# Initialize modules for users not using bash as a default shell
modules_home=${MODULESHOME:-"/usr/share/lmod/lmod"}
if test -e ${modules_home}/init/bash
then
  . ${modules_home}/init/bash
fi

set -o errexit
set -o nounset

option=${1:-""}
hostname="$(hostname)"
cluster=${hostname//[0-9]/}
project_dir="$(git rev-parse --show-toplevel)"
if [[ $? -eq 1 ]]
then
    project_dir="$(pwd)"
fi

spec=${SPEC:-""}
modules=${MODULES:-""}
# NOTE: No modules will be explicitly unloaded or purged. Obviously,
# loading a new compiler will trigger the auto-unload of the existing
# compiler module (and all the other side-effects wrt mpi, etc), but
# no explicit action is taken by this script.

build_root=${BUILD_ROOT:-""}
hostconfig=${HOST_CONFIG:-""}
job_unique_id=${CI_JOB_ID:-""}
spack_upstream=${UPSTREAM:-""}

prefix=""

# Setup the module environment
if [[ -n "${modules}" ]]
then
    echo "Loading modules: \"${modules}\""
    module load ${modules}
fi

# The ${project_dir}/.uberenv_config.json file has paths relative to
# the toplevel, and things break if uberenv isn't invoked there. This
# seems easier than trying to shim something into that file. Even if
# taking a path through this script that avoids uberenv, the rest of
# it likely has some subtle dependence on this anyway.
cd ${project_dir}

if [[ -d /dev/shm ]]
then
    prefix="/dev/shm/${hostname}"
    if [[ -z ${job_unique_id} ]]; then
        job_unique_id=manual_job_$(date +%s)
        while [[ -d ${prefix}-${job_unique_id} ]] ; do
            sleep 1
            job_unique_id=manual_job_$(date +%s)
        done
    fi

    prefix="${prefix}-${job_unique_id}"
    mkdir -p ${prefix}
fi

# Dependencies
date
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test started"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

if [[ "${option}" != "--build-only" && "${option}" != "--test-only" ]]
then
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ -z ${spec} ]]
    then
        echo "SPEC is undefined, aborting..."
        exit 1
    fi

    prefix_opt=""

    if [[ -d /dev/shm ]]
    then
        prefix_opt="--prefix=${prefix}"

        # We force Spack to put all generated files (cache and configuration of
        # all sorts) in a unique location so that there can be no collision
        # with existing or concurrent Spack.
        spack_user_cache="${prefix}/spack-user-cache"
        export SPACK_DISABLE_LOCAL_CONFIG=""
        export SPACK_USER_CACHE_PATH="${spack_user_cache}"
        mkdir -p ${spack_user_cache}
    fi

    upstream_opt=""
    if [[ -n "${spack_upstream}" && -d ${spack_upstream} ]]
    then
        upstream_opt="--upstream=${spack_upstream}"
    fi

    spack_file="${project_dir}/scripts/spack/environments/${cluster}/spack.yaml"
    env_file_opt=""
    if [[ -e "${spack_file}" ]]
    then
        env_file_opt="--spack-env-file=${spack_file}"
    fi

    ${project_dir}/scripts/uberenv/uberenv.py \
                  --spec="${spec}" ${env_file_opt} ${upstream_opt} ${prefix_opt}

fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Dependencies Built"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date

# Host config file
if [[ -z ${hostconfig} ]]
then
    # If no host config file was provided, we assume it was generated.
    # This means we are looking of a unique one in project dir.
    hostconfigs=( $( ls "${project_dir}/"*.cmake ) )
    if [[ ${#hostconfigs[@]} == 1 ]]
    then
        hostconfig_path=${hostconfigs[0]}
        echo "Found host config file: ${hostconfig_path}"
    elif [[ ${#hostconfigs[@]} == 0 ]]
    then
        echo "No result for: ${project_dir}/*.cmake"
        echo "Spack generated host-config not found."
        exit 1
    else
        echo "More than one result for: ${project_dir}/*.cmake"
        echo "${hostconfigs[@]}"
        echo "Please specify one with HOST_CONFIG variable"
        exit 1
    fi
else
    # Using provided host-config file.
    hostconfig_path="${project_dir}/${hostconfig}"
fi

hostconfig=$(basename ${hostconfig_path})

# Build Directory
if [[ -z ${build_root} ]]
then
    if [[ -d /dev/shm ]]
    then
        build_root="${prefix}"
    else
        build_root="$(pwd)"
    fi
else
    build_root="${build_root}"
fi

build_root="$(pwd)"
build_dir="${build_root}/build_${hostconfig//.cmake/}"
install_dir="${build_root}/install_${hostconfig//.cmake/}"

cmake_exe=`grep 'CMake executable' ${hostconfig_path} | cut -d ':' -f 2 | xargs`

# Build
if [[ "${option}" != "--deps-only" && "${option}" != "--test-only" ]]
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Host-config: ${hostconfig_path}"
    echo "~~~~~ Build Dir:   ${build_dir}"
    echo "~~~~~ Project Dir: ${project_dir}"
    echo "~~~~~ Install Dir: ${install_dir}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo ""
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building DiHydrogen"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Map CPU core allocations
    declare -A core_counts=(["lassen"]=40 ["corona"]=32 ["tioga"]=32 ["pascal"]=36 ["catalyst"]=24)

    # If building, then delete everything first
    # NOTE: 'cmake --build . -j core_counts' attempts to reduce individual build resources.
    #       If core_counts does not contain hostname, then will default to '-j ', which should
    #       use max cores.
    rm -rf ${build_dir} 2>/dev/null
    mkdir -p ${build_dir} && cd ${build_dir}

    date

    generator="Unix Makefiles"
    if command -v ninja > /dev/null;
    then
        generator="Ninja"
    fi
    echo "Using ${generator} generator."

    $cmake_exe \
        -G "${generator}" \
        -C ${hostconfig_path} \
        -DCMAKE_INSTALL_PREFIX=${install_dir} \
        ${project_dir}
    if ! $cmake_exe --build . -j ${core_counts[$cluster]}
    then
        echo "ERROR: compilation failed, building with verbose output..."
        $cmake_exe --build . --verbose -j 1
    else
        $cmake_exe --install .
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ DiHydrogen Built"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    date

fi

# Test
if [[ "${option}" != "--build-only" ]] && grep -q -i "H2_ENABLE_TESTS.*ON" ${hostconfig_path}
then
    date
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Testing DiHydrogen"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    if [[ ! -d ${build_dir} ]]
    then
        echo "ERROR: Build directory not found : ${build_dir}" && exit 1
    fi

    cd ${build_dir}

    # Run tests
    ctest_exe=$(dirname ${cmake_exe})/ctest
    date
    ${ctest_exe} \
        --output-on-failure \
        --output-junit \
        ${project_dir}/${hostname}_junit.xml |& tee tests_output.txt
    date

    no_test_str="No tests were found!!!"
    if [[ "$(tail -n 1 tests_output.txt)" == "${no_test_str}" ]]
    then
        echo "ERROR: No tests were found" && exit 1
    fi

    if grep -q "Errors while running CTest" ./tests_output.txt
    then
        echo "ERROR: failure(s) while running CTest" && exit 1
    fi

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Dihydrogen Tests Complete"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    date

fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test completed"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
date
