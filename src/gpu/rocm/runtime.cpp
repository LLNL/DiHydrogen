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

#include <hip/hip_runtime.h>
#include <rocm_smi/rocm_smi.h>

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

#define H2_CHECK_RSMI(rsmi_call)                                               \
    do                                                                         \
    {                                                                          \
        auto const check_rsmi_status = (rsmi_call);                            \
        if (check_rsmi_status != RSMI_STATUS_SUCCESS)                          \
        {                                                                      \
            std::ostringstream oss;                                            \
            oss << "RSMI call failed "                                         \
                << "(" << __FILE__ << ":" << __LINE__ << "): "                 \
                << "[" << rsmi_status_to_string(check_rsmi_status) << "] \""   \
                << rsmi_status_description(check_rsmi_status) << "\"";         \
            throw std::runtime_error(oss.str());                               \
        }                                                                      \
    } while (0)

namespace
{

// This is overkill for the portion of the API we end up using, but
// the copy/paste/sed that generated this worked just as well on all
// cases as it would have on the 3-5 that matter. And I didn't have to
// tabulate the list of the ones that matter.
static char const* rsmi_status_description(rsmi_status_t const status)
{
    switch (status)
    {
    case RSMI_STATUS_SUCCESS: return "Operation was successful";
    case RSMI_STATUS_INVALID_ARGS: return "Passed in arguments are not valid";
    case RSMI_STATUS_NOT_SUPPORTED:
        return "The requested information or action is not available for the "
               "given input, on the given system";
    case RSMI_STATUS_FILE_ERROR:
        return "Problem accessing a file. This may because the operation is "
               "not supported by the Linux kernel version running on the "
               "executing machine";
    case RSMI_STATUS_PERMISSION:
        return "Permission denied/EACCESS file error. Many functions require "
               "root access to run.";
    case RSMI_STATUS_OUT_OF_RESOURCES:
        return "Unable to acquire memory or other resource";
    case RSMI_STATUS_INTERNAL_EXCEPTION:
        return "An internal exception was caught";
    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
        return "The provided input is out of allowable or safe range";
    case RSMI_STATUS_INIT_ERROR:
        return "An error occurred when rsmi initializing internal data "
               "structures";
    case RSMI_STATUS_NOT_YET_IMPLEMENTED:
        return "The requested function has not yet been implemented in the "
               "current system for the current devices";
    case RSMI_STATUS_NOT_FOUND: return "An item was searched for but not found";
    case RSMI_STATUS_INSUFFICIENT_SIZE:
        return "Not enough resources were available for the operation";
    case RSMI_STATUS_INTERRUPT:
        return "An interrupt occurred during execution of function";
    case RSMI_STATUS_UNEXPECTED_SIZE:
        return "An unexpected amount of data was read";
    case RSMI_STATUS_NO_DATA: return "No data was found for a given input";
    case RSMI_STATUS_UNEXPECTED_DATA:
        return "The data read or provided to function is not what was expected";
    case RSMI_STATUS_BUSY:
        return "A resource or mutex could not be acquired because it is "
               "already being used";
    case RSMI_STATUS_REFCOUNT_OVERFLOW:
        return "An internal reference counter exceeded INT32_MAX";
    case RSMI_STATUS_UNKNOWN_ERROR: return "An unknown error occurred";
    default: return "<UNKNOWN RSMI ERROR CODE>";
    }
}

static char const* rsmi_status_to_string(rsmi_status_t const status)
{
    switch (status)
    {
    case RSMI_STATUS_SUCCESS: return "RSMI_STATUS_SUCCESS";
    case RSMI_STATUS_INVALID_ARGS: return "RSMI_STATUS_INVALID_ARGS";
    case RSMI_STATUS_NOT_SUPPORTED: return "RSMI_STATUS_NOT_SUPPORTED";
    case RSMI_STATUS_FILE_ERROR: return "RSMI_STATUS_FILE_ERROR";
    case RSMI_STATUS_PERMISSION: return "RSMI_STATUS_PERMISSION";
    case RSMI_STATUS_OUT_OF_RESOURCES: return "RSMI_STATUS_OUT_OF_RESOURCES";
    case RSMI_STATUS_INTERNAL_EXCEPTION:
        return "RSMI_STATUS_INTERNAL_EXCEPTION";
    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
        return "RSMI_STATUS_INPUT_OUT_OF_BOUNDS";
    case RSMI_STATUS_INIT_ERROR: return "RSMI_STATUS_INIT_ERROR";
    case RSMI_STATUS_NOT_YET_IMPLEMENTED:
        return "RSMI_STATUS_NOT_YET_IMPLEMENTED";
    case RSMI_STATUS_NOT_FOUND: return "RSMI_STATUS_NOT_FOUND";
    case RSMI_STATUS_INSUFFICIENT_SIZE: return "RSMI_STATUS_INSUFFICIENT_SIZE";
    case RSMI_STATUS_INTERRUPT: return "RSMI_STATUS_INTERRUPT";
    case RSMI_STATUS_UNEXPECTED_SIZE: return "RSMI_STATUS_UNEXPECTED_SIZE";
    case RSMI_STATUS_NO_DATA: return "RSMI_STATUS_NO_DATA";
    case RSMI_STATUS_UNEXPECTED_DATA: return "RSMI_STATUS_UNEXPECTED_DATA";
    case RSMI_STATUS_BUSY: return "RSMI_STATUS_BUSY";
    case RSMI_STATUS_REFCOUNT_OVERFLOW: return "RSMI_STATUS_REFCOUNT_OVERFLOW";
    case RSMI_STATUS_UNKNOWN_ERROR: return "RSMI_STATUS_UNKNOWN_ERROR";
    default: return "<UNKNOWN RSMI ERROR CODE>";
    }
}

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

static void error_terminate(char const* const msg)
{
    H2_GPU_ERROR(msg);
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
        H2_GPU_WARN("Could not guess local rank; setting device 0.");
        return 0;
    }

    if (lsize < 0)
    {
        H2_GPU_WARN("Could not guess local size; setting device 0.");
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

    error_terminate("More local ranks than (visible) GPUs.");
    return -1;
}

static void set_reasonable_default_gpu()
{
    h2::gpu::set_gpu(get_reasonable_default_gpu_id());
}

static std::string get_device_name_by_pci_bus(int const pci_bus)
{
    uint32_t dev_id = 0, ndevices = 0;
    uint64_t bdfid = 0;
    H2_CHECK_RSMI(rsmi_num_monitor_devices(&ndevices));
    for (; dev_id < ndevices; ++dev_id)
    {
        H2_CHECK_RSMI(rsmi_dev_pci_id_get(dev_id, &bdfid));
        // Just need the bus ID.
        int const rsmi_pci = static_cast<int>((bdfid >> 8) & 0xff);
        if (rsmi_pci == pci_bus)
            break;
    }

    if (dev_id == ndevices)
    {
        throw std::runtime_error(
            "Could not find device with matching PCI bus ID");
    }

    char name[256];
    H2_CHECK_RSMI(rsmi_dev_name_get(dev_id, name, 256));
    return name;
}

static void log_gpu_info(int const gpu_id)
{
    int pci;
    H2_CHECK_HIP(
        hipDeviceGetAttribute(&pci, hipDeviceAttributePciBusId, gpu_id));

    try
    {
        H2_CHECK_RSMI(rsmi_init(0));
        auto const name = get_device_name_by_pci_bus(pci);
        H2_GPU_TRACE("GPU ID {}: name=\"{}\", pci={:#04x}", gpu_id, name, pci);
    }
    catch (std::runtime_error const& e)
    {
        // Report the error; just log the PCI bus and move on.
        H2_GPU_ERROR("Non-fatal ROCm-SMI error: {}", e.what());
        H2_GPU_TRACE("GPU ID {}: pci={:#04x}", gpu_id, pci);
    }

    // Ignoring the error code here because we're done with RSMI
    // anyway. Either rsmi_init failed above, in which case this will
    // just return RSMI_STATUS_INIT_ERROR, or RSMI is active and needs
    // to be shut down, in which case this shouldn't fail but we don't
    // really care if it does because we don't need RSMI anyway and we
    // tried our darnedest, and that's what really counts. If the
    // whole GPU ecosystem is broken, some H2_CHECK_HIP will die
    // shortly anyway.
    static_cast<void>(rsmi_shut_down());
}

} // namespace

int h2::gpu::num_gpus()
{
    int count;
    H2_CHECK_HIP(hipGetDeviceCount(&count));
    return count;
}

int h2::gpu::current_gpu()
{
    int dev;
    H2_CHECK_HIP(hipGetDevice(&dev));
    return dev;
}

void h2::gpu::set_gpu(int id)
{
    H2_GPU_TRACE("setting device to id={}", id);
    H2_CHECK_HIP(hipSetDevice(id));
}

void h2::gpu::init_runtime()
{
    if (!initialized_)
    {
        H2_GPU_TRACE("initializing gpu runtime");
        H2_CHECK_HIP(hipInit(0));
        H2_GPU_TRACE("found {} devices", num_gpus());
        set_reasonable_default_gpu();
        initialized_ = true;
    }
    else
    {
        H2_GPU_TRACE("H2 GPU already initialized; current gpu={}", current_gpu());
    }
    log_gpu_info(current_gpu());
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

hipStream_t h2::gpu::make_stream()
{
    hipStream_t stream;
    H2_CHECK_HIP(hipStreamCreate(&stream));
    H2_GPU_TRACE("created stream {}", (void*) stream);
    return stream;
}

hipStream_t h2::gpu::make_stream_nonblocking()
{
    hipStream_t stream;
    H2_CHECK_HIP(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    H2_GPU_TRACE("created non-blocking stream {}", (void*) stream);
    return stream;
}

void h2::gpu::destroy(hipStream_t stream)
{
    H2_GPU_TRACE("destroy stream {}", (void*) stream);
    H2_CHECK_HIP(hipStreamDestroy(stream));
}

hipEvent_t h2::gpu::make_event()
{
    hipEvent_t event;
    H2_CHECK_HIP(hipEventCreateWithFlags(&event, hipEventDisableSystemFence));
    H2_GPU_TRACE("created event {}", (void*) event);
    return event;
}

hipEvent_t h2::gpu::make_event_notiming()
{
    hipEvent_t event;
    H2_CHECK_HIP(hipEventCreateWithFlags(
        &event, hipEventDisableTiming | hipEventDisableSystemFence));
    H2_GPU_TRACE("created non-timing event {}", (void*) event);
    return event;
}

void h2::gpu::destroy(hipEvent_t const event)
{
    H2_GPU_TRACE("destroy event {}", (void*) event);
    H2_CHECK_HIP(hipEventDestroy(event));
}

void h2::gpu::sync()
{
    H2_GPU_TRACE("synchronizing gpu");
    H2_CHECK_HIP(hipDeviceSynchronize());
}

void h2::gpu::sync(hipEvent_t event)
{
    H2_GPU_TRACE("synchronizing event {}", (void*) event);
    H2_CHECK_HIP(hipEventSynchronize(event));
}

void h2::gpu::sync(hipStream_t stream)
{
    H2_GPU_TRACE("synchronizing stream {}", (void*) stream);
    H2_CHECK_HIP(hipStreamSynchronize(stream));
}
