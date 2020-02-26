#pragma once

#include <cuda_runtime.h>

void spin_device(cudaStream_t s, int giga_cycles);
