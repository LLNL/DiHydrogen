#pragma once

#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

void spin_gpu(int spin_ms, cudaStream_t stream);
