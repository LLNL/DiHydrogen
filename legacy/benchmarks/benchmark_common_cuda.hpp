#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

void spin_gpu(int spin_ms, cudaStream_t stream);
