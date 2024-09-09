////////////////////////////////////////////////////////////////////////////////
// Copyright 2019-2022 Lawrence Livermore National Security, LLC and other
// DiHydrogen Project Developers. See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: Apache-2.0
////////////////////////////////////////////////////////////////////////////////
#include "distconv/util/util_rocm.hpp"

#include "h2/gpu/memory_utils.hpp"
#include "h2/gpu/runtime.hpp"

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>

namespace distconv
{
namespace util
{
void check_for_device_runtime_error()
{
  DISTCONV_CHECK_HIP(hipGetLastError());
}

int get_number_of_gpus()
{
  if (char* const env = getenv("TENSOR_NUM_GPUS"))
  {
    std::cerr << "Number of GPUs set by TENSOR_NUM_GPUS" << std::endl;
    return atoi(env);
  }
  else
    return h2::gpu::num_gpus();
}

int get_local_rank()
{
  char* env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env)
    env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env)
    env = getenv("SLURM_LOCALID");
  if (!env)
    env = getenv("FLUX_TASK_LOCAL_ID");
  if (!env)
  {
    std::cerr << "Can't determine local rank" << std::endl;
    abort();
  }
  return atoi(env);
}

int get_local_size()
{
  char* env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env)
    env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env)
    env = getenv("SLURM_TASKS_PER_NODE");
  if (!env)
  {
    std::cerr << "Can't determine local size" << std::endl;
    abort();
  }
  return atoi(env);
}

int choose_gpu()
{
  int const num_gpus = get_number_of_gpus();
  int const local_rank = get_local_rank();
  int const local_size = get_local_size();
  if (num_gpus < local_size)
  {
    std::cerr << "Warning: Number of GPUs, " << num_gpus
              << " is smaller than the number of local MPI ranks, "
              << local_size << std::endl;
  }
  int const gpu = local_rank % num_gpus;
  return gpu;
}

std::ostream& operator<<(std::ostream& os, hipPitchedPtr const& p)
{
  return os << "hipPitchedPtr(" << p.ptr << ", " << p.pitch << ", " << p.xsize
            << ", " << p.ysize << ")";
}

std::ostream& operator<<(std::ostream& os, hipPos const& p)
{
  return os << "hipPos(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream& operator<<(std::ostream& os, hipMemcpy3DParms const& p)
{
  os << "hipMemcpy3DParms(srcPtr: " << p.srcPtr << ", dstPtr: " << p.dstPtr
     << ", srcPos: " << p.srcPos << ", dstPos: " << p.dstPos << ")";
  return os;
}

hipError_t hip_malloc(void** ptr,
                      size_t const size,
                      char const* const file_name,
                      int const linum)
{
  // Report only when file_name is given and the size is larger than
  // one Mib by default
  char* log_env = std::getenv("DISTCONV_LOG_HIP_MALLOC");
  if (log_env && file_name)
  {
    int threshold = 0;
    try
    {
      threshold = std::stoi(std::string(log_env));
    }
    catch (std::invalid_argument const&)
    {}
    int const size_in_mb = size / (1024 * 1024);
    if (size_in_mb >= threshold)
    {
      util::MPIPrintStreamInfo() << "hipMalloc of " << size_in_mb << " MiB at "
                                 << file_name << ":" << linum;
    }
  }
  auto const st = hipMalloc(ptr, size);
  if (st != hipSuccess)
  {
    auto const& [available, total] = h2::gpu::mem_info();
    util::MPIPrintStreamError()
      << "Allocation of " << size << " bytes ("
      << size / 1024.0 / 1024.0 / 1024.0 << " GiB) failed. " << available
      << " bytes (" << available / 1024.0 / 1024.0 / 1024.0
      << " GiB) available out of " << total << " bytes ( "
      << total / 1024.0 / 1024.0 / 1024.0 << " GiB).";
    DISTCONV_CHECK_HIP(hipGetLastError());
  }
  return st;
}

void wait_stream(hipStream_t const master,
                 hipStream_t* const followers,
                 int const num_followers)
{
  hipEvent_t const ev = internal::RuntimeHIP::get_event();
  bool event_recorded = false;
  for (int i = 0; i < num_followers; ++i)
  {
    hipStream_t follower = followers[i];
    if (master == follower)
      continue;
    if (!event_recorded)
    {
      DISTCONV_CHECK_HIP(hipEventRecord(ev, master));
      event_recorded = true;
    }
    DISTCONV_CHECK_HIP(hipStreamWaitEvent(follower, ev, 0));
  }
}

void wait_stream(hipStream_t const master, hipStream_t follower)
{
  wait_stream(master, &follower, 1);
}

void sync_stream(hipStream_t const s1, hipStream_t const s2)
{
  if (s1 == s2)
    return;
  hipEvent_t ev1 = internal::RuntimeHIP::get_event(0);
  hipEvent_t ev2 = internal::RuntimeHIP::get_event(1);
  DISTCONV_CHECK_HIP(hipEventRecord(ev1, s1));
  DISTCONV_CHECK_HIP(hipEventRecord(ev2, s2));
  DISTCONV_CHECK_HIP(hipStreamWaitEvent(s2, ev1, 0));
  DISTCONV_CHECK_HIP(hipStreamWaitEvent(s1, ev2, 0));
}

}  // namespace util
}  // namespace distconv

namespace
{
std::pair<int, int> get_stream_priority_bounds()
{
  int least_priority, greatest_priority;
  DISTCONV_CHECK_HIP(
    hipDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
  return {least_priority, greatest_priority};
}
}  // namespace

hipStream_t distconv::util::create_priority_stream()
{
  static int const greatest_priority = get_stream_priority_bounds().second;
  hipStream_t s;
  DISTCONV_CHECK_HIP(
    hipStreamCreateWithPriority(&s, hipStreamNonBlocking, greatest_priority));
  return s;
}
