#!/usr/bin/env bash

# Usage:
# Change to the directory where tensor test binaries are located and run this script. 

set -u

if (( $# > 0 )); then
	LAUNCH_METHOD=${1,,}
else
	LAUNCH_METHOD=""
fi

NUM_PROCS=4
PX=2
PY=2

echo "NUM_PROCS: ${NUM_PROCS}"

export MV2_USE_CUDA=1
####################################################
TEST_PROC=(test_tensor)
TEST_CUDA=(test_tensor_cuda)
TEST_MPI=(test_tensor_mpi test_tensor_mpi_copy)
TEST_MPI_CUDA=(test_tensor_mpi_cuda test_tensor_mpi_cuda_copy
			   test_tensor_mpi_cuda_shuffle
			   test_tensor_mpi_cuda_algorithms
			   test_halo_exchange_cuda)
####################################################
failed_tests=""
####################################################
function get_hostfile_slurm() {
	hostfile=$1
	srun -n $NUM_PROCS hostname > $hostfile
}

function get_hostfile_lsf() {
	hostfile=$1
	cat $LSB_DJOB_HOSTFILE > $hostfile
	while [ $(wc -l $hostfile | awk '{print $1}') -lt $NUM_PROCS ]; do
		cat $hostfile $hostfile > $hostfile.tmp
		mv $hostfile.tmp $hostfile
	done
}

function get_hostfile() {
	if [ -v LSB_DJOB_HOSTFILE ]; then
		get_hostfile_lsf $*
	elif [ -v SLURM_JOB_UID ]; then
		get_hostfile_slurm $*
	fi
}

HOSTFILE=hosts
get_hostfile $HOSTFILE
####################################################
function run_slurm() {
	SLURM_HOSTFILE=$HOSTFILE srun -n $NUM_PROCS --distribution=arbitrary $*
}

function run_mvapich() {
	mpirun_rsh -np $NUM_PROCS -hostfile $HOSTFILE $*
}

function mpi_run() {
	if [[ $LAUNCH_METHOD = slurm ||
			  -v SLURM_JOB_UID ]]; then
		run_slurm $*
	elif [[ $LAUNCH_METHOD = mvapich ||
				$(type mpirun_rsh) =~ "mpirun_rsh is " ]]; then
		run_mvapich $*
	else
		echo "Unknown launch method: $LAUNCH_METHOD"
		exit 1
	fi
}
		
####################################################
function run_test_proc() {
	echo "Running proc tests"
	local failed=""
	for t in ${TEST_PROC[*]}; do
		echo "Running $t"
		./$t
		if [ $? -ne 0 ]; then
			echo "Test $t failed"
			failed+="$t "
		fi
	done
	if [[ -z $failed ]]; then
		echo "Proc tests completed successfully"
	else
		echo "Proc failed tests: $failed"
		failed_tests+="$failed "
	fi
}
####################################################
function run_test_cuda() {
	echo "Running CUDA tests"
	local failed=""
	for t in ${TEST_CUDA[*]}; do
		echo "Running $t"
		./$t
		if [ $? -ne 0 ]; then
			echo "Test $t failed"
			failed+="$t "
		fi
	done
	if [[ -z $failed ]]; then
		echo "CUDA tests completed successfully"
	else
		echo "CUDA failed tests: $failed"
		failed_tests+="$failed "
	fi
}
####################################################
function run_test_mpi() {
	echo "Running MPI tests"
	local failed=""
	for t in ${TEST_MPI[*]}; do
		echo "Running $t"
		local args=""
		if [[ $t = test_tensor_mpi_copy ]]; then
			args+="$PX $PY"
		fi
		mpi_run ./$t $args
		if [ $? -ne 0 ]; then
			echo "Test $t failed"
			failed+="$t "
		fi
	done
	if [[ -z $failed ]]; then
		echo "MPI tests completed successfully"
	else
		echo "MPI failed tests: $failed"
		failed_tests+="$failed "
	fi
}
####################################################
function run_test_mpi_cuda() {
	echo "Running MPI-CUDA tests"
	local failed=""
	for t in ${TEST_MPI_CUDA[*]}; do
		echo "Running $t"
		local args=""
		if [[ $t = test_tensor_mpi_cuda_copy ||
				  $t = test_tensor_mpi_cuda_shuffle ||
				  $t = test_halo_exchange_cuda
			]]; then
			args+="$PX $PY"
		fi
		mpi_run ./$t $args
		if [ $? -ne 0 ]; then
			echo "Test $t failed"
			failed+="$t "
		fi
	done
	if [[ -z $failed ]]; then
		echo "MPI-CUDA tests completed successfully"
	else
		echo "MPI-CUDA failed tests: $failed"
		failed_tests+="$failed "
	fi
}
####################################################

run_test_proc
run_test_cuda
run_test_mpi
run_test_mpi_cuda

echo -e "\n--------------------------------\n"
echo -e "All tests completed\n"

if [[ -z $failed_tests ]]; then
	echo "No failure"
else
	echo "Failed tests: $failed_tests"
fi
