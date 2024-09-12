#!/bin/bash

################################################################################
## Copyright 2019-2024 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

# This script looks for a known rank variable that is set and writes
# its value to stdout. If none are found, nothing is printed and 1 is
# returned.

declare -a rank_vars
rank_vars=(
    FLUX_TASK_RANK
    SLURM_PROCID
    PMI_RANK
    MV2_COMM_WORLD_RANK
    OMPI_COMM_WORLD_RANK)

for var in "${rank_vars[@]}"
do
    if [[ -n "${!var}" ]]
    then
        echo "${!var}"
        exit 0
    fi
done
exit 1
