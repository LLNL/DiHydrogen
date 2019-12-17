#include "distconv/util/util_cuda.hpp"

#include <string>

namespace distconv {
namespace util {

int get_number_of_gpus() {
  int num_gpus = 0;
  char *env = getenv("TENSOR_NUM_GPUS");
  if (env) {
    std::cout << "Number of GPUs set by TENSOR_NUM_GPUS\n";
    num_gpus = atoi(env);
  } else {
    DISTCONV_CHECK_CUDA(cudaGetDeviceCount(&num_gpus));    
  }
  return num_gpus;
}

int get_local_rank() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = getenv("SLURM_LOCALID");
  if (!env) {
    std::cerr << "Can't determine local rank\n";
    abort();
  }
  return atoi(env);
}

int get_local_size() {
  char *env = getenv("MV2_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("OMPI_COMM_WORLD_LOCAL_SIZE");
  if (!env) env = getenv("SLURM_TASKS_PER_NODE");  
  if (!env) {
    std::cerr << "Can't determine local size\n";
    abort();
  }
  return atoi(env);
}

int choose_gpu() {
  int num_gpus = get_number_of_gpus();
  int local_rank = get_local_rank();
  int local_size = get_local_size();
  if (num_gpus < local_size) {
    std::cerr << "Warning: Number of GPUs, " << num_gpus
              << " is smaller than the number of local MPI ranks, "
              << local_size << "\n";
  }
  int gpu = local_rank % num_gpus;
  return gpu;
}

std::ostream &operator<<(std::ostream &os, const cudaPitchedPtr &p) {
  return os << "cudaPitchedPtr(" << p.ptr << ", " << p.pitch << ", " << p.xsize << ", " << p.ysize << ")";
}

std::ostream &operator<<(std::ostream &os, const cudaPos &p) {
  return os << "cudaPos(" << p.x << ", " << p.y << ", " << p.z << ")";
}

std::ostream &operator<<(std::ostream &os, const cudaMemcpy3DParms &p) {
  os << "cudaMemcpy3DParms(srcPtr: " << p.srcPtr << ", dstPtr: " << p.dstPtr
     << ", srcPos: " << p.srcPos << ", dstPos: " << p.dstPos
     << ")";
  return os;
}

cudaError_t cuda_malloc(void **ptr, size_t size,
                        const char *file_name, int linum) {
  // Report only when file_name is given and the size is larger than
  // one Mib by default
  char *log_env = std::getenv("DISTCONV_LOG_CUDA_MALLOC");
  if (log_env && file_name) {
    int threshold = 0;
    try {
      threshold = std::stoi(std::string(log_env));
    } catch (std::invalid_argument) {
    }
    int size_in_mb = size / (1024 * 1024);
    if (size_in_mb >= threshold) {
      util::MPIPrintStreamInfo()
          << "cudaMalloc of " << size_in_mb
          << " MiB at " << file_name << ":" << linum;
    }
  }
  auto st = cudaMalloc(ptr, size);
  if (st != cudaSuccess) {
    size_t available;
    size_t total;
    DISTCONV_CHECK_CUDA(cudaMemGetInfo(&available, &total));
    util::MPIPrintStreamError()
        << "Allocation of " << size << " bytes ("
        << size / 1024.0 / 1024.0 / 1024.0 << " GiB) failed. "
        << available << " bytes ("
        << available / 1024.0 / 1024.0 / 1024.0
        << " GiB) available out of " << total << " bytes ( "
        << total / 1024.0 / 1024.0 / 1024.0 << " GiB).";
    DISTCONV_CHECK_CUDA(cudaGetLastError());
  }
  return st;
}

void wait_stream(cudaStream_t master, cudaStream_t *followers,
                 int num_followers) {
  cudaEvent_t ev = internal::RuntimeCUDA::get_event();
  bool event_recorded = false;
  for (int i = 0; i < num_followers; ++i) {
    cudaStream_t follower = followers[i];
    if (master == follower) continue;
    if (!event_recorded) {
      DISTCONV_CHECK_CUDA(cudaEventRecord(ev, master));
      event_recorded = true;
    }
    DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(follower, ev, 0));
  }
}

void wait_stream(cudaStream_t master, cudaStream_t follower) {
  wait_stream(master, &follower, 1);
}

void sync_stream(cudaStream_t s1, cudaStream_t s2) {
  if (s1 == s2) return;
  cudaEvent_t ev1 = internal::RuntimeCUDA::get_event(0);
  cudaEvent_t ev2 = internal::RuntimeCUDA::get_event(1);
  DISTCONV_CHECK_CUDA(cudaEventRecord(ev1, s1));
  DISTCONV_CHECK_CUDA(cudaEventRecord(ev2, s2));
  DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(s2, ev1, 0));
  DISTCONV_CHECK_CUDA(cudaStreamWaitEvent(s1, ev2, 0));
}

cudaStream_t create_priority_stream() {
  int least_priority, greatest_priority;
  cudaStream_t s;
  DISTCONV_CHECK_CUDA(cudaDeviceGetStreamPriorityRange(
      &least_priority, &greatest_priority));
  DISTCONV_CHECK_CUDA(cudaStreamCreateWithPriority(
      &s, cudaStreamNonBlocking, greatest_priority));
  return s;
}

} // namespace util
} // namespace distconv
