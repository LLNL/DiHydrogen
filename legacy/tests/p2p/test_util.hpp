#pragma once

#include <cstdlib>

inline int get_local_rank() {
  char *env = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
  if (!env) env = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (!env) env = std::getenv("SLURM_LOCALID");
  if (!env) {
    std::cerr << "Can't determine local rank\n";
    std::abort();
  }
  return std::atoi(env);
}

#define TEST_RUN(test_call) do {                \
  int x = test_call;                            \
  if (x != 0) {                                 \
    std::cerr << "Test failed: "                \
              << #test_call << std::endl;       \
  } } while (0)
