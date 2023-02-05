////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "h2/gpu/runtime.hpp"

#include "h2/gpu/logger.hpp"

#include <cstdlib>
#include <cstring>
#include <iostream> // FIXME: Eventually, Logger.hpp

#include <cuda_runtime.h>

// Note: The behavior of functions in this file may be impacted by the
// following environment variables:
//
//   - FLUX_TASK_LOCAL_ID
//   - SLURM_LOCALID
//   - SLURM_NTASKS_PER_NODE
//   - OMPI_COMM_WORLD_LOCAL_RANK
//   - OMPI_COMM_WORLD_LOCAL_SIZE
//   - MV2_COMM_WORLD_LOCAL_RANK
//   - MV2_COMM_WORLD_LOCAL_SIZE
//   - MPI_LOCALRANKID
//   - MPI_LOCALNRANKS
//
// The user may set the following to any string that matches "[^0].*"
// to effect certain behaviors, as described below:
//
//   - H2_SELECT_DEVICE_0: If set to a truthy value, every MPI rank
//                         will call hipSetDevice(0). This could save
//                         you from a bad binding (e.g., if using
//                         mpibind) or it could cause oversubscription
//                         (e.g., if you also set
//                         ROCR_VISIBLE_DEVICES=0 or something).
//
//   - H2_SELECT_DEVICE_RR: If set to a truthy value, every MPI rank
//                          will call
//                          hipSetDevice(local_rank%num_visible_gpus). This
//                          option is considered AFTER
//                          H2_SELECT_DEVICE_0, so if both are set,
//                          device 0 will be selected.
//
// The behavior is undefined if the value of the H2_* variables
// differs across processes in one MPI universe.

namespace
{

// There are a few cases here:
//
// mpibind=off: See all GPUs/GCDs on a node.
// mpibind=on: See ngpus/local_rnks GPUs.
//   -> ngpus > local_rnks: Many choices.
//   -> ngpus = local_rnks: Pick rank 0.
//   -> ngpus < local_rnks: Oversubscription.
//
// We should have reasonable behavior for all cases (which might just
// be to raise an error).
bool initialized_ = false;

static int guess_local_rank() noexcept
{
    // Start with launchers, then poke the MPI libs
    char const* env = std::getenv("FLUX_TASK_LOCAL_ID");
    if (!env)
        env = std::getenv("SLURM_LOCALID");
    if (!env)
        env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"); // Open-MPI
    if (!env)
        env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK"); // MVAPICH2
    if (!env)
        env = std::getenv("MPI_LOCALRANKID"); // MPICH
    return (env ? std::atoi(env) : -1);
}

static int guess_local_size() noexcept
{
    // Let's assume that ranks are balanced across nodes in flux-land...
    if (char const* flux_size = std::getenv("FLUX_JOB_SIZE"))
    {
        char const* nnodes = std::getenv("FLUX_JOB_NNODES");
        if (nnodes)
        {
            auto const int_nnodes = std::atoi(nnodes);
            return (std::atoi(flux_size) + int_nnodes - 1) / int_nnodes;
        }
    }

    char const* env = std::getenv("SLURM_NTASKS_PER_NODE");
    if (!env)
        env = std::getenv("OMPI_COMM_WORLD_LOCAL_SIZE"); // Open-MPI
    if (!env)
        env = std::getenv("MV2_COMM_WORLD_LOCAL_SIZE"); // MVAPICH2
    if (!env)
        env = std::getenv("MPI_LOCALNRANKS"); // MPICH
    return (env ? std::atoi(env) : -1);
}

// Unset -> false
// Empty -> false
// 0 -> false
static bool check_bool_cstr(char const* const str)
{
    return (str && std::strlen(str) && str[0] != '0');
}

static bool force_device_zero() noexcept
{
    return check_bool_cstr(std::getenv("H2_SELECT_DEVICE_0"));
}

static bool force_round_robin() noexcept
{
    return check_bool_cstr(std::getenv("H2_SELECT_DEVICE_RR"));
}

static void warn(char const* const msg)
{
    std::clog << msg << std::endl;
}

static void warn(std::string const& msg)
{
    std::clog << msg << std::endl;
}

static void error(char const* const msg)
{
    std::clog << msg << std::endl;
    std::terminate();
}

static void error(std::string const& msg)
{
    std::clog << msg << std::endl;
    std::terminate();
}

// This just uses the HIP runtime and/or user-provided environment
// variables. A more robust solution might tap directly into HWLOC or
// something of that nature. We should also look into whether we can
// (easily) access more information about the running job, such as the
// REAL number of GPUs on a node (since the runtime is swayed by env
// variables) or even just whether or not a job has been launched with
// mpibind enabled.
static int get_reasonable_default_gpu_id() noexcept
{
    // Check if the user has requested device 0.
    if (force_device_zero())
        return 0;

    int const lrank = guess_local_rank();
    int const lsize = guess_local_size();
    if (lrank < 0)
    {
        warn("Could not guess local rank; setting device 0.");
        return 0;
    }

    if (lsize < 0)
    {
        warn("Could not guess local size; setting device 0.");
        return 0;
    }

    // Force the round-robin if it's been requested.
    int const ngpus = h2::gpu::num_gpus();
    if (force_round_robin())
        return lrank % ngpus;

    // At this point, we can just branch based on the relationship of
    // ngpus and nlocal_rnks. If we risk oversubscription, we can
    // error out at this point.
    if (lsize <= ngpus)
        return lrank;

    error("More local ranks than (visible) GPUs.");
    return -1;
}

static void set_reasonable_default_gpu()
{
    h2::gpu::set_gpu(get_reasonable_default_gpu_id());
}

} // namespace

int h2::gpu::num_gpus()
{
    int count;
    H2_CHECK_CUDA(cudaGetDeviceCount(&count));
    return count;
}

int h2::gpu::current_gpu()
{
    int dev;
    H2_CHECK_CUDA(cudaGetDevice(&dev));
    return dev;
}

void h2::gpu::set_gpu(int id)
{
    H2_GPU_TRACE("setting device to id={}", id);
    H2_CHECK_CUDA(cudaSetDevice(id));
}

void h2::gpu::init_runtime()
{
    if (initialized_)
        return;

    H2_GPU_TRACE("initializing gpu runtime");
    H2_GPU_TRACE("found {} devices", num_gpus());
    set_reasonable_default_gpu();
    initialized_ = true;
}

void h2::gpu::finalize_runtime()
{
    if (!initialized_)
        return;

    H2_GPU_TRACE("finalizing gpu runtime");
    initialized_ = false;
}

bool h2::gpu::runtime_is_initialized()
{
    return initialized_;
}

bool h2::gpu::runtime_is_finalized()
{
    return !initialized_;
}

cudaStream_t h2::gpu::make_stream()
{
    cudaStream_t stream;
    H2_CHECK_CUDA(cudaStreamCreate(&stream));
    H2_GPU_TRACE("created stream {}", (void*) stream);
    return stream;
}

cudaStream_t h2::gpu::make_stream_nonblocking()
{
    cudaStream_t stream;
    H2_CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    H2_GPU_TRACE("created non-blocking stream {}", (void*) stream);
    return stream;
}

void h2::gpu::destroy(cudaStream_t const stream)
{
    H2_GPU_TRACE("destroy stream {}", (void*) stream);
    H2_CHECK_CUDA(cudaStreamDestroy(stream));
}

cudaEvent_t h2::gpu::make_event()
{
    cudaEvent_t event;
    H2_CHECK_CUDA(cudaEventCreate(&event));
    H2_GPU_TRACE("created event {}", (void*) event);
    return event;
}

cudaEvent_t h2::gpu::make_event_notiming()
{
    cudaEvent_t event;
    H2_CHECK_CUDA(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    H2_GPU_TRACE("created non-timing event {}", (void*) event);
    return event;
}

void h2::gpu::destroy(cudaEvent_t const event)
{
    H2_GPU_TRACE("destroy event {}", (void*) event);
    H2_CHECK_CUDA(cudaEventDestroy(event));
}

void h2::gpu::sync()
{
    H2_GPU_TRACE("synchronizing gpu");
    H2_CHECK_CUDA(cudaDeviceSynchronize());
}

void h2::gpu::sync(cudaEvent_t event)
{
    H2_GPU_TRACE("synchronizing event {}", (void*) event);
    H2_CHECK_CUDA(cudaEventSynchronize(event));
}

void h2::gpu::sync(cudaStream_t stream)
{
    H2_GPU_TRACE("synchronizing stream {}", (void*) stream);
    H2_CHECK_CUDA(cudaStreamSynchronize(stream));
}
